import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset

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

        os.makedirs(
            spectral_dir,
            exist_ok=True
        )

        ##################################################
        # Download only first 230 cubes
        ##################################################

        if download:

            existing = [
                f for f in os.listdir(
                    spectral_dir
                )
                if f.endswith(".mat")
            ]

            if len(existing) < total_images:

                print(
                    f"Downloading "
                    f"{total_images} cubes..."
                )

                repo_files = list_repo_files(
                    "mhmdjouni/arad_hsdb",
                    repo_type="dataset"
                )

                mat_files = sorted([
                    f
                    for f in repo_files
                    if (
                        f.endswith(".mat")
                        and
                        "NTIRE2020_Train_Spectral"
                        in f
                    )
                ])

                mat_files = mat_files[
                    :total_images
                ]

                for file in mat_files:

                    hf_hub_download(
                        repo_id=
                        "mhmdjouni/arad_hsdb",

                        repo_type=
                        "dataset",

                        filename=file,

                        local_dir=root_dir,

                        local_dir_use_symlinks=False
                    )

                print(
                    "Download complete"
                )

        ##################################################
        # Gather local files
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

        files = files[:total_images]

        print(
            f"Using {len(files)} cubes"
        )

        ##################################################
        # Split
        ##################################################

        if train:

            self.files = files[
                :train_images
            ]

        else:

            self.files = files[
                train_images:
            ]

        print(
            f"{'Train' if train else 'Val'}: "
            f"{len(self.files)} cubes"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(
        self,
        idx
    ):

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

        cube = np.transpose(
            cube,
            (2, 0, 1)
        )

        cube = torch.from_numpy(
            cube
        ).float()

        cube = torch.from_numpy(
            cube
        ).float()

        cube = F.interpolate(
            cube.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return cube
