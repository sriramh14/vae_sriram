# download_arad.py
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mhmdjouni/arad_hsdb",
    repo_type="dataset",
    allow_patterns=[
        "NTIRE2020_Train_Spectral/*.mat"
    ],
    local_dir="data"
)

print("Download complete")


#To verify the download
root_dir = "data/NTIRE2020_Train_Spectral"

mat_files = sorted([
    os.path.join(root_dir, f)
    for f in os.listdir(root_dir)
    if f.endswith(".mat")
])

print("Number of files:", len(mat_files))

train_files = mat_files[:368]   # 80%
val_files   = mat_files[368:]   # 20%

print(len(train_files))
print(len(val_files))
