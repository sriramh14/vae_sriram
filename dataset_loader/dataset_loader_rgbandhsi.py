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
        download=True
    ):

        self.cube_key = cube_key

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
        # Download dataset
        ##################################################

        if download:

            repo_files = list_repo_files(
                "mhmdjouni/arad_hsdb",
                repo_type="dataset"
            )

            spectral_files = sorted([
                f
                for f in repo_files
                if (
                    f.endswith(".mat")
                    and
                    "NTIRE2020_Train_Spectral"
                    in f
                )
            ])[:total_images]

            rgb_files = sorted([
                f
                for f in repo_files
                if (
                    f.endswith(".jpg")
                    and
                    "NTIRE2020_Train_RealWorld"
                    in f
                )
            ])[:total_images]

            print(
                f"Downloading "
                f"{len(spectral_files)} HSI files "
                f"and "
                f"{len(rgb_files)} RGB files..."
            )

            for file in spectral_files:

                hf_hub_download(
                    repo_id="mhmdjouni/arad_hsdb",
                    repo_type="dataset",
                    filename=file,
                    local_dir=root_dir,
                    local_dir_use_symlinks=False
                )

            for file in rgb_files:

                hf_hub_download(
                    repo_id="mhmdjouni/arad_hsdb",
                    repo_type="dataset",
                    filename=file,
                    local_dir=root_dir,
                    local_dir_use_symlinks=False
                )

            print("Download complete")

        ##################################################
        # Build paired samples
        ##################################################

        spectral_files = sorted([
            f
            for f in os.listdir(
                spectral_dir
            )
            if f.endswith(".mat")
        ])[:total_images]

        self.pairs = []

        for mat_name in spectral_files:

            stem = os.path.splitext(
                mat_name
            )[0]

            rgb_name = stem + ".jpg"

            rgb_path = os.path.join(
                rgb_dir,
                rgb_name
            )

            if not os.path.exists(
                rgb_path
            ):
                continue

            self.pairs.append(
                (
                    os.path.join(
                        spectral_dir,
                        mat_name
                    ),
                    rgb_path
                )
            )

        print(
            f"Found {len(self.pairs)} paired samples"
        )

        ##################################################
        # Train / Val split
        ##################################################

        if train:

            self.pairs = self.pairs[
                :train_images
            ]

        else:

            self.pairs = self.pairs[
                train_images:
            ]

        print(
            f"{'Train' if train else 'Val'}: "
            f"{len(self.pairs)} samples"
        )

    def __len__(self):

        return len(
            self.pairs
        )

    def __getitem__(
        self,
        idx
    ):

        hsi_path, rgb_path = self.pairs[idx]

        ##################################################
        # Load HSI
        ##################################################

        mat = sio.loadmat(
            hsi_path
        )

        hsi = mat[
            self.cube_key
        ].astype(
            np.float32
        )

        if hsi.max() > 1:

            hsi /= hsi.max()

        hsi = np.transpose(
            hsi,
            (2, 0, 1)
        )

        hsi = torch.from_numpy(
            hsi
        ).float()

        hsi = F.interpolate(
            hsi.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        ##################################################
        # Load RGB
        ##################################################

        rgb = Image.open(
            rgb_path
        ).convert("RGB")

        rgb = np.array(
            rgb,
            dtype=np.float32
        ) / 255.0

        rgb = np.transpose(
            rgb,
            (2, 0, 1)
        )

        rgb = torch.from_numpy(
            rgb
        ).float()

        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        ##################################################
        # Return pair
        ##################################################

        return rgb, hsi
