# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from env.rewards import *
import math

import numpy as np
import copy
import torch_pruning as tp


class ChannelPruningEnv:
    """
    Env for channel pruning search using torch-pruning
    """
    def __init__(self, model, checkpoint, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):
        # default setting
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]

        # save options
        self.model = model
        self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound
        self.use_real_val = args.use_real_val
        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root
        self.export_model = export_model
        self.use_new_input = use_new_input

        # record the pruning‐order names once
        self.layer_names = [
            name
            for name, m in self.model.named_modules()
            if isinstance(m, nn.Conv2d)
            ]

        # Save the original model and checkpoint using deepcopy
        self.original_model = copy.deepcopy(model)
        self.original_checkpoint = copy.deepcopy(checkpoint)

        self.model = copy.deepcopy(self.original_model)
        self.checkpoint = copy.deepcopy(self.original_checkpoint)

        # prepare data
        self._init_data()

        # torch-pruning: build dependency graph and pruner
        example_inputs, _ = next(iter(self.train_loader))
        example_inputs = example_inputs.cuda()
        self.example_inputs = example_inputs

        ignore_names = []
        # Find the last fc in the classifier (for various models like MobileNetV2)
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                # Estimate the last fc if out_features == n_class
                if hasattr(self, "n_class") and m.out_features == self.n_class:
                    ignore_names.append(name)
        # Also include conv1
        if hasattr(self.model, "conv1"):
            for name, m in self.model.named_modules():
                if m is self.model.conv1:
                    ignore_names.append(name)

        ignored_layers = []
        for name, m in self.model.named_modules():
            if name in ignore_names:
                ignored_layers.append(m)
        self.ignored_layers = ignored_layers

        self.pruner = tp.pruner.BasePruner(
            self.model,
            self.example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(p=1),
            pruning_ratio=1 - self.preserve_ratio,
            ignored_layers=self.ignored_layers,
            round_to=self.channel_round,
        )

        # get prunable groups for state embedding
        self.prunable_groups = list(self.pruner.DG.get_all_groups(
            ignored_layers=self.ignored_layers,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear]
        ))
        self.n_prunable_layer = len(self.prunable_groups)

        # build embedding (static part)
        self._build_state_embedding()

        # Save original channels for each prunable group
        self.org_channels = []
        for group in self.prunable_groups:
            dep, _ = group[0]
            m = dep.target.module
            if isinstance(m, nn.Conv2d):
                self.org_channels.append(m.out_channels)
            elif isinstance(m, nn.Linear):
                self.org_channels.append(m.out_features)
            else:
                self.org_channels.append(0)  # fallback, should not happen

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        self.org_model_size = sum(p.numel() for p in self.model.parameters())
        print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
        self.org_flops, _ = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops

        self.reward = eval(args.reward)
        self.best_reward = -math.inf
        self.best_strategy = None

    def step(self, action):
        # action: pruning ratio for the current group
        if self.cur_ind >= self.n_prunable_layer:
            raise RuntimeError("All layers already pruned. Call reset().")

        group = self.prunable_groups[self.cur_ind]
        # action: float in [0,1], 1 means keep all, 0 means prune all
        action = float(action)
        action = np.clip(action, self.lbound, self.rbound)
        pruning_ratio = 1 - float(action)
        dep, idxs = group[0]
        n_ch = len(idxs)
        n_prune = int(n_ch * pruning_ratio)
        if n_prune < 1:
            n_prune = 1
        # group-level importance (see tp_README.md)
        imp = self.pruner.importance(group)
        # importance has the same length as idxs
        prune_indices = np.argsort(imp.cpu().numpy())[:n_prune].tolist()
        group.prune(idxs=[idxs[i] for i in prune_indices])

        self.strategy.append(action)
        self.cur_ind += 1

        if self.cur_ind == self.n_prunable_layer:
            # all pruned, evaluate
            current_flops, _ = tp.utils.count_ops_and_params(self.model, self.example_inputs)
            acc = self._validate(self.val_loader, self.model)
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            reward = self.reward(self, acc, current_flops)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                # Collect channel counts after pruning
                pruned_channels = []
                for group in self.prunable_groups:
                    dep, _ = group[0]
                    m = dep.target.module
                    if isinstance(m, nn.Conv2d):
                        pruned_channels.append(m.out_channels)
                    elif isinstance(m, nn.Linear):
                        pruned_channels.append(m.out_features)
                    else:
                        pruned_channels.append(0)
                prGreen('New best policy: {}'.format([float(t) for t in self.best_strategy]))
                prGreen('Channels after pruning: {}'.format(pruned_channels))
                import os
                # Save to export_path directory if set
                if hasattr(self, "export_path") and self.export_path is not None:
                    export_dir = os.path.dirname(self.export_path)
                    if not export_dir:
                        export_dir = "."
                    policy_path = os.path.join(export_dir, "best_policy.txt")
                    with open(policy_path, "w") as f:
                        f.write("policy," + ",".join([str(float(t)) for t in self.best_strategy]) + "\n")
                        f.write("channels," + ",".join([str(int(c)) for c in pruned_channels]) + "\n")
                # Always save to output(logs) directory if available
                if hasattr(self.args, "output") and self.args.output is not None:
                    policy_path2 = os.path.join(self.args.output, "best_policy.txt")
                    with open(policy_path2, "w") as f:
                        f.write("policy," + ",".join([str(float(t)) for t in self.best_strategy]) + "\n")
                        f.write("channels," + ",".join([str(int(c)) for c in pruned_channels]) + "\n")
            obs = self.layer_embedding[self.cur_ind - 1, :].copy()
            done = True
            if self.export_model:
                torch.save(self.model, self.export_path)
                return None, None, None, None
            return obs, reward, done, info_set

        # next state
        obs = self.layer_embedding[self.cur_ind, :].copy()
        done = False
        reward = 0
        info_set = None
        return obs, reward, done, info_set

    def reset(self):
        # Assign a new deepcopy of the entire model
        self.model = copy.deepcopy(self.original_model)
        self.model.load_state_dict(self.original_checkpoint)
        self.cur_ind = 0
        self.strategy = []

        ignore_names = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                if hasattr(self, "n_class") and m.out_features == self.n_class:
                    ignore_names.append(name)
        if hasattr(self.model, "conv1"):
            for name, m in self.model.named_modules():
                if m is self.model.conv1:
                    ignore_names.append(name)

        ignored_layers = []
        for name, m in self.model.named_modules():
            if name in ignore_names:
                ignored_layers.append(m)
        self.ignored_layers = ignored_layers
        # re-init pruner and prunable_groups
        self.pruner = tp.pruner.BasePruner(
            self.model,
            self.example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(p=2),
            pruning_ratio=1 - self.preserve_ratio,
            ignored_layers=self.ignored_layers,
            round_to=self.channel_round,
        )
        self.prunable_groups = list(self.pruner.DG.get_all_groups(
            ignored_layers=self.ignored_layers,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear]
        ))
        obs = self.layer_embedding[0].copy()
        return obs

    def set_export_path(self, path):
        self.export_path = path

    def _init_data(self):
        # split the train set into train + val
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        self.n_class = n_class  # Save n_class
        if self.use_real_val:  # use the real val set for eval, which is actually wrong
            print('*** USE REAL VALIDATION SET!')

    def _build_state_embedding(self):
        # build the static part of the state embedding using prunable_groups
        layer_embedding = []
        for i, group in enumerate(self.prunable_groups):
            dep, _ = group[0]
            m = dep.target.module
            this_state = []
            if isinstance(m, nn.Conv2d):
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)
                this_state.append(m.out_channels)
                this_state.append(m.stride[0])
                this_state.append(m.kernel_size[0])
                this_state.append(np.prod(m.weight.size()))
            elif isinstance(m, nn.Linear):
                this_state.append(i)
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)
                this_state.append(m.out_features)
                this_state.append(0)
                this_state.append(1)
                this_state.append(np.prod(m.weight.size()))
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)
        self.layer_embedding = layer_embedding

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.acc_metric == 'acc1':
            return top1.avg
        elif self.acc_metric == 'acc5':
            return top5.avg
        else:
            raise NotImplementedError
