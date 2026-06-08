class ARADDataset(Dataset):

    def __init__(
        self,
        root_dir,
        patch_size=128,
        train=True,
        train_images=300,
        patches_per_image=20,
        cube_key="cube"
    ):

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.cube_key = cube_key

        files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".mat")
        ])

        if train:
            self.files = files[:train_images]
        else:
            self.files = files[train_images:]

    def __len__(self):

        return (
            len(self.files)
            * self.patches_per_image
        )

    def __getitem__(self, idx):

        file_idx = (
            idx
            // self.patches_per_image
        )

        mat = sio.loadmat(
            self.files[file_idx]
        )

        cube = mat[self.cube_key]

        cube = cube.astype(
            np.float32
        )

        if cube.max() > 1:
            cube /= cube.max()

        H, W, C = cube.shape

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
            top:top+ps,
            left:left+ps,
            :
        ]

        cube = np.transpose(
            cube,
            (2,0,1)
        )

        return torch.from_numpy(
            cube
        ).float()
