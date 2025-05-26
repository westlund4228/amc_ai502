import torch

class SensitiveLayerFinder:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.activations = {}

    def _hook(self, name):
        def fn(_, __, output):
            # flatten each feature-map and compute its rank
            fmap = output.detach().cpu()
            per_filter_ranks = []
            # for each channel j, compute rank per sample and then average
            for j in range(fmap.size(1)):
                # fmap[:, j] has shape [batch, H, W]
                ranks_j = torch.linalg.matrix_rank(fmap[:, j])   # -> tensor of shape [batch]
                avg_rank_j = ranks_j.float().mean().item()        # one scalar
                per_filter_ranks.append(avg_rank_j)
        
            self.activations[name].append(per_filter_ranks)
        return fn

    def compute_rank_expectations(self):
        # register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.activations[name] = []
                module.register_forward_hook(self._hook(name))

        # run through a few batches
        with torch.no_grad():
            for i, (imgs, _) in enumerate(self.dataloader):
                if i >= 10: break
                imgs = imgs.to(self.device)
                self.model(imgs)

        # average ranks across samples
        rank_exps = {}
        for name, lists in self.activations.items():
            # lists is a list of [per_filter_ranks] for each batch
            mean_ranks = torch.tensor(lists).mean(dim=0).tolist()
            rank_exps[name] = mean_ranks
        return rank_exps

    def identify_sensitive_layers(self, rank_exps):
        sensitive = []
        for name, ranks in rank_exps.items():
            # compute threshold T_i per Eq. (3) and count filters > T_i
            T_i = torch.tensor(ranks).mean().item()  # placeholder for variance‐based formula
            high = sum(r > T_i for r in ranks)
            if high > len(ranks) / 2:
                sensitive.append(name)
        return sensitive
