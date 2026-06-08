import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset_loader.Dataset_loader import ARADDataset
from models.HSIBetaVAE import HSIBetaVAE
from Eval.MRAE import mrae_loss
from Eval.KL_Loss import kl_loss
from Eval.SAM import sam_loss



##################################################
# CONFIG
##################################################

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_EPOCHS = 100

BATCH_SIZE = 1

NUM_WORKERS = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LATENT_CHANNELS = 8

BETA_MAX = 1e-5

WARMUP_EPOCHS = 20

CHECKPOINT_DIR = "checkpoints"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


##################################################
# DATASETS
##################################################

train_dataset = ARADDataset(
    train=True,
    train_images=200,
    total_images=230,
    cube_key="cube"
)

val_dataset = ARADDataset(
    train=False,
    train_images=200,
    total_images=230,
    cube_key="cube",
    download=False
)

##################################################
# DATALOADERS
##################################################

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


##################################################
# MODEL
##################################################

model = HSIBetaVAE(
    in_channels=31,
    latent_channels=LATENT_CHANNELS
).to(DEVICE)


##################################################
# OPTIMIZER
##################################################

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


##################################################
# VALIDATION
##################################################

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0
    total_mrae = 0
    total_sam = 0

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

        total_loss += loss.item()
        total_mrae += mrae.item()
        total_sam += sam.item()

    n = len(val_loader)

    return (
        total_loss / n,
        total_mrae / n,
        total_sam / n
    )


##################################################
# TRAINING LOOP
##################################################

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

    running_loss = 0
    running_kl = 0

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

        running_loss += loss.item()
        running_kl += kl.item()

    train_loss = (
        running_loss
        / len(train_loader)
    )

    train_kl = (
        running_kl
        / len(train_loader)
    )

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

    ##################################################
    # SAVE BEST MODEL
    ##################################################

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
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
        }

        torch.save(
            checkpoint,
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

        print(
            "Saved best checkpoint"
        )


##################################################
# SAVE FINAL MODEL
##################################################

torch.save(
    model.decoder.state_dict(),
    os.path.join(
        CHECKPOINT_DIR,
        "decoder_final.pth"
    )
)

torch.save(
    model.encoder.state_dict(),
    os.path.join(
        CHECKPOINT_DIR,
        "encoder_final.pth"
    )
)

print("Training complete")
