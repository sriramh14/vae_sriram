import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset_loader.dataset_loader_rgbandhsi import ARADDataset

from models.HSIencoder import HSIEncoder
from models.HSIDecoder import HSIDecoder
from models.RGBEncoder import RGBEncoder

from Eval.MRAE import mrae_loss
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

BATCH_SIZE = 4

NUM_WORKERS = 2

LR = 1e-4

LATENT_CHANNELS = 8

CHECKPOINT_DIR = "checkpoints_rgb"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


##################################################
# DATASET
##################################################

train_dataset = ARADDataset(
    train=True
)

val_dataset = ARADDataset(
    train=False
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


##################################################
# MODELS
##################################################

hsi_encoder = HSIEncoder(
    in_channels=31,
    latent_channels=LATENT_CHANNELS
).to(DEVICE)

hsi_decoder = HSIDecoder(
    latent_channels=LATENT_CHANNELS,
    out_channels=31
).to(DEVICE)

rgb_encoder = RGBEncoder(
    in_channels = 3,
    latent_channels=LATENT_CHANNELS
).to(DEVICE)


##################################################
# LOAD PRETRAINED VAE
##################################################

vae_ckpt = torch.load(
    "checkpoints/best_model.pth",
    map_location=DEVICE
)


hsi_encoder.load_state_dict(
    vae_ckpt["encoder"]
)

hsi_decoder.load_state_dict(
    vae_ckpt["decoder"]
)
if "model_state_dict" in vae_ckpt:
    vae_ckpt = vae_ckpt["model_state_dict"]

print("Loaded pretrained VAE")


##################################################
# FREEZE HSI MODELS
##################################################

hsi_encoder.eval()
hsi_decoder.eval()

for p in hsi_encoder.parameters():
    p.requires_grad = False

for p in hsi_decoder.parameters():
    p.requires_grad = False


##################################################
# OPTIMIZER
##################################################

optimizer = torch.optim.AdamW(
    rgb_encoder.parameters(),
    lr=LR
)


##################################################
# VALIDATION
##################################################

@torch.no_grad()
def validate():

    rgb_encoder.eval()

    total_loss = 0
    total_mrae = 0
    total_sam = 0

    for batch in val_loader:

        rgb = batch["rgb"].to(DEVICE)
        hsi = batch["hsi"].to(DEVICE)

        gt_mu, _ = hsi_encoder(hsi)

        pred_z = rgb_encoder(rgb)

        pred_hsi = hsi_decoder(pred_z)

        latent_loss = F.mse_loss(
            pred_z,
            gt_mu
        )

        recon_loss = F.l1_loss(
            pred_hsi,
            hsi
        )

        loss = (
            latent_loss
            + 0.1 * recon_loss
        )

        mrae = mrae_loss(
            pred_hsi,
            hsi
        )

        sam = sam_loss(
            pred_hsi,
            hsi
        )

        total_loss += loss.item()
        total_mrae += mrae.item()
        total_sam += sam.item()

    rgb_encoder.train()

    n = len(val_loader)

    return (
        total_loss / n,
        total_mrae / n,
        total_sam / n
    )


##################################################
# TRAINING
##################################################

best_mrae = 1e9

for epoch in range(NUM_EPOCHS):

    rgb_encoder.train()

    running_loss = 0

    for batch in train_loader:

        rgb, hsi = batch

        rgb = rgb.to(DEVICE)
        hsi = hsi.to(DEVICE)

        with torch.no_grad():

            gt_mu, _ = hsi_encoder(hsi)

        pred_z = rgb_encoder(rgb)

        pred_hsi = hsi_decoder(pred_z)

        latent_loss = F.mse_loss(
            pred_z,
            gt_mu
        )

        recon_loss = F.l1_loss(
            pred_hsi,
            hsi
        )

        loss = (
            latent_loss
            + 0.1 * recon_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss /
        len(train_loader)
    )

    val_loss, val_mrae, val_sam = validate()

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f} | "
        f"MRAE {val_mrae:.6f} | "
        f"SAM {val_sam:.6f}"
    )

    if val_mrae < best_mrae:

        best_mrae = val_mrae

        torch.save(
            rgb_encoder.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "rgb_encoder_best.pth"
            )
        )

        print(
            f"Saved best model "
            f"(MRAE={val_mrae:.6f})"
        )
