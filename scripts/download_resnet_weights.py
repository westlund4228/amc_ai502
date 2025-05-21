import os
import zipfile
import argparse
import requests
from tqdm import tqdm
import shutil

def download_zip(url, save_path):
    if os.path.exists(save_path):
        print(f"Zip file already exists at {save_path}, skipping download.")
        return

    print(f"Downloading from {url}")
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(save_path, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Download failed or incomplete")
    print("Download complete.")

def extract_model(zip_path, model_name, extract_dir, keep_all=False):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        target_file = f"state_dicts/{model_name}.pt"

        if target_file not in all_files:
            raise FileNotFoundError(f"{target_file} not found in zip archive.")

        os.makedirs(extract_dir, exist_ok=True)

        if keep_all:
            print("Extracting all files (keep_all=True)...")
            zip_ref.extractall(extract_dir)
        else:
            print(f"Extracting only {target_file}...")
            zip_ref.extract(target_file, extract_dir)

        extracted_path = os.path.join(extract_dir, target_file)
        final_path = os.path.join("models", "state_dicts", f"{model_name}.pt")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.rename(extracted_path, final_path)
        print(f"Moved {model_name}.pt to models/state_dicts/")

    if not keep_all:
        shutil.rmtree(extract_dir)
        os.remove(zip_path)
        print("Removed zip and temp directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", help="Model name to extract")
    parser.add_argument("--keep", action="store_true", help="Keep all weights from zip (default: delete others)")
    args = parser.parse_args()

    ZIP_URL = "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ZIP_PATH = os.path.join(PROJECT_ROOT, "models", "state_dicts.zip")
    TEMP_EXTRACT_DIR = os.path.join(PROJECT_ROOT, "models", "temp_extract")

    download_zip(ZIP_URL, ZIP_PATH)
    extract_model(ZIP_PATH, args.model, TEMP_EXTRACT_DIR, keep_all=args.keep)
