import os
import numpy as np
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from huggingface_hub import (
    list_repo_files,
    hf_hub_download
)


class ARADDataset(Dataset):

    def __init__(
        self,
        root_dir="data",
        train=True,
        train_images=200,
        total_images=230,
        cube_key="cube",
        download=True,
        image_size=256
    ):

        self.cube_key = cube_key
        self.image_size = image_size

        spectral_dir = os.path.join(
            root_dir,
            "NTIRE2020_Train_Spectral"
        )

        rgb_dir = os.path.join(
            root_dir,
            "NTIRE2020_Train_RealWorld"
        )

        os.makedirs(
            spectral_dir,
            exist_ok=True
        )

        os.makedirs(
            rgb_dir,
            exist_ok=True
        )

        ##################################################
        # Download files
        ##################################################

        if download:

            existing_files = [
                f for f in os.listdir(spectral_dir)
                if (f.endswith(".mat") or f.endswith(".jpg"))
            ]

            if len(existing_mats) < 2 * total_images:

                print(
                    f"Downloading "
                    f"{2*total_images} samples..."
                )

                repo_files = list_repo_files(
                    "mhmdjouni/arad_hsdb",
                    repo_type="dataset"
                )

                mat_files = sorted([
                    f for f in repo_files
                    if (
                        f.endswith(".mat")
                        and
                        "NTIRE2020_Train_Spectral" in f
                    )
                ])[:total_images]

                rgb_files = sorted([
                    f for f in repo_files
                    if (
                        f.endswith(".jpg")
                        and
                        "NTIRE2020_Train_RealWorld" in f
                    )
                ])[:total_images]

                files_to_download = (
                    mat_files +
                    rgb_files
                )

                for file in files_to_download:

                    hf_hub_download(
                        repo_id="mhmdjouni/arad_hsdb",
                        repo_type="dataset",
                        filename=file,
                        local_dir=root_dir,
                        local_dir_use_symlinks=False
                    )

                print(
                    "Download complete"
                )

        ##################################################
        # Match RGB ↔ HSI files
        ##################################################

        mat_files = sorted([
            os.path.join(
                spectral_dir,
                f
            )
            for f in os.listdir(
                spectral_dir
            )
            if f.endswith(".mat")
        ])[:total_images]

        pairs = []

        for mat_path in mat_files:

            stem = os.path.splitext(
                os.path.basename(mat_path)
            )[0]

            jpg_path = os.path.join(
                rgb_dir,
                stem + ".jpg"
            )

            if os.path.exists(jpg_path):

                pairs.append(
                    (
                        mat_path,
                        jpg_path
                    )
                )

        print(
            f"Found {len(pairs)} paired samples"
        )

        ##################################################
        # Split
        ##################################################

        if train:

            self.samples = pairs[
                :train_images
            ]

        else:

            self.samples = pairs[
                train_images:
            ]

        print(
            f"{'Train' if train else 'Val'}: "
            f"{len(self.samples)} samples"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        mat_path, jpg_path = (
            self.samples[idx]
        )

        ##################################################
        # Load HSI
        ##################################################

        mat = sio.loadmat(
            mat_path
        )

        cube = mat[
            self.cube_key
        ].astype(
            np.float32
        )

        if cube.max() > 1:

            cube /= cube.max()

        cube = np.transpose(
            cube,
            (2, 0, 1)
        )

        hsi = torch.from_numpy(
            cube
        ).float()

        hsi = F.interpolate(
            hsi.unsqueeze(0),
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        ##################################################
        # Load RGB
        ##################################################

        rgb = Image.open(
            jpg_path
        ).convert(
            "RGB"
        )

        rgb = np.array(
            rgb,
            dtype=np.float32
        ) / 255.0

        rgb = torch.from_numpy(
            rgb
        ).permute(
            2,
            0,
            1
        )

        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return {
            "rgb": rgb,      # [3,H,W]
            "hsi": hsi       # [31,H,W]
        }
