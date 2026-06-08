import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import HSIBetaVAE
from dataset import ARADDataset
from losses import (
    mrae_loss,
    sam_loss,
    kl_loss
)

DEVICE = "cuda"

ROOT_DIR = (
    "data/NTIRE2020_Train_Spectral"
)

BATCH_SIZE = 1

NUM_WORKERS = 2

NUM_EPOCHS = 100

LATENT_CHANNELS = 8

BETA_MAX = 1e-5

WARMUP_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "checkpoints"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



all_files = sorted([
    os.path.join(ROOT_DIR, f)
    for f in os.listdir(ROOT_DIR)
    if f.endswith(".mat")
])

print(
    f"Total spectral cubes: {len(all_files)}"
)

split_idx = int(
    0.8 * len(all_files)
)

train_files = all_files[:split_idx]

val_files = all_files[split_idx:]

print(
    f"Train scenes: {len(train_files)}"
)

print(
    f"Val scenes: {len(val_files)}"
)


train_dataset = ARADDataset(
    files=train_files,
    cube_key="cube"   # replace if needed
)

val_dataset = ARADDataset(
    files=val_files,
    cube_key="cube"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = HSIBetaVAE(
    in_channels=31,
    latent_channels=8
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

beta_max = 1e-5
warmup_epochs = 20
@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0

    running_mrae = 0

    running_sam = 0

    for hsi in val_loader:

        hsi = hsi.to(
            DEVICE,
            non_blocking=True
        )

        recon, mu, logvar, _ = model(hsi)

        l1 = F.l1_loss(
            recon,
            hsi
        )

        mrae = mrae_loss(
            recon,
            hsi
        )

        sam = sam_loss(
            recon,
            hsi
        )

        loss = (
            0.5 * l1
            + 0.4 * mrae
            + 0.1 * sam
        )

        running_loss += loss.item()

        running_mrae += mrae.item()

        running_sam += sam.item()

    n = len(val_loader)

    return (
        running_loss / n,
        running_mrae / n,
        running_sam / n
    )

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    model.train()

    beta = (
        BETA_MAX
        * min(
            epoch / WARMUP_EPOCHS,
            1.0
        )
    )

    train_loss = 0

    train_kl = 0

    for hsi in train_loader:

        hsi = hsi.to(
            DEVICE,
            non_blocking=True
        )

        recon, mu, logvar, z = model(hsi)

        l1 = F.l1_loss(
            recon,
            hsi
        )

        mrae = mrae_loss(
            recon,
            hsi
        )

        sam = sam_loss(
            recon,
            hsi
        )

        kl = kl_loss(
            mu,
            logvar
        )

        recon_loss = (
            0.5 * l1
            + 0.4 * mrae
            + 0.1 * sam
        )

        loss = (
            recon_loss
            + beta * kl
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_kl += kl.item()

    train_loss /= len(train_loader)

    train_kl /= len(train_loader)

    val_loss, val_mrae, val_sam = validate()

    print(
        f"[{epoch+1:03d}/{NUM_EPOCHS}] "
        f"Train={train_loss:.5f} "
        f"Val={val_loss:.5f} "
        f"MRAE={val_mrae:.5f} "
        f"SAM={val_sam:.5f} "
        f"KL={train_kl:.5f} "
        f"Beta={beta:.2e}"
    )

    ####################################
    # SAVE BEST MODEL
    ####################################

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,

                "vae":
                    model.state_dict(),

                "encoder":
                    model.encoder.state_dict(),

                "decoder":
                    model.decoder.state_dict(),

                "optimizer":
                    optimizer.state_dict(),

                "val_loss":
                    val_loss
            },
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

        print(
            "Saved best checkpoint"
        )

print("Training complete")
