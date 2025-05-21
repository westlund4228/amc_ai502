import urllib.request
from pathlib import Path

dropbox_url = "https://www.dropbox.com/scl/fi/u68wsofs5ufce9pwh920n/mobilenetv1_cifar.pth.tar?rlkey=q0i7zu5719gvle93r16xl3amx&st=hfdyw10p&dl=1"
local_path = Path("models/state_dict/mobilenetv1_cifar.pth.tar")
local_path.parent.mkdir(parents=True, exist_ok=True)

urllib.request.urlretrieve(dropbox_url, local_path)

print(f"Model saved to {local_path}")