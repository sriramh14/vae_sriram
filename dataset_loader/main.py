class ARAD300Dataset(Dataset):

    def __init__(
        self,
        patch_size=128
    ):

        self.dataset = load_dataset(
            "mhmdjouni/arad_hsdb",
            split="train[:300]"
        )

        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        cube = None

        for value in sample.values():

            if isinstance(value, np.ndarray):

                if value.ndim == 3:
                    cube = value
                    break

        if cube is None:
            raise RuntimeError(
                "Could not find hyperspectral cube"
            )

        cube = cube.astype(np.float32)

        if cube.max() > 1:
            cube /= cube.max()

        # Convert HWC -> CHW if needed
        if cube.shape[-1] == 31:

            cube = np.transpose(
                cube,
                (2, 0, 1)
            )

        C, H, W = cube.shape

        ps = self.patch_size

        top = random.randint(
            0,
            H - ps
        )

        left = random.randint(
            0,
            W - ps
        )

        cube = cube[
            :,
            top:top+ps,
            left:left+ps
        ]

        return torch.from_numpy(
            cube
        ).float()
