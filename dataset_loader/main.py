import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset

from huggingface_hub import (
    snapshot_download
)


class ARADDataset(Dataset):

    def __init__(
        self,
        root_dir="data",
        train=True,
        train_images=300,
        cube_key="cube",
        download=True
    ):

        self.cube_key = cube_key

        spectral_dir = os.path.join(
            root_dir,
            "NTIRE2020_Train_Spectral"
        )

        ##################################################
        # Download dataset if missing
        ##################################################

        if download and (
            not os.path.exists(spectral_dir)
            or len(os.listdir(spectral_dir)) == 0
        ):

            print(
                "Downloading ARAD spectral cubes..."
            )

            snapshot_download(
                repo_id="mhmdjouni/arad_hsdb",
                repo_type="dataset",
                allow_patterns=[
                    "NTIRE2020_Train_Spectral/*.mat"
                ],
                local_dir=root_dir
            )

            print(
                "Download complete."
            )

        ##################################################
        # Collect all MAT files
        ##################################################

        files = sorted([
            os.path.join(
                spectral_dir,
                f
            )
            for f in os.listdir(
                spectral_dir
            )
            if f.endswith(".mat")
        ])

        print(
            f"Found {len(files)} "
            f"spectral cubes"
        )

        ##################################################
        # Train / Validation split
        ##################################################

        if train:

            self.files = files[:train_images]

        else:

            self.files = files[train_images:]

        print(
            f"{'Train' if train else 'Val'} "
            f"samples: {len(self.files)}"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        mat = sio.loadmat(
            self.files[idx]
        )

        cube = mat[
            self.cube_key
        ]

        cube = cube.astype(
            np.float32
        )

        if cube.max() > 1:

            cube /= cube.max()

        ##################################################
        # HWC -> CHW
        ##################################################

        cube = np.transpose(
            cube,
            (2, 0, 1)
        )

        cube = torch.from_numpy(
            cube
        ).float()

        return cube
