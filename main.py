device = "cuda"

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

for epoch in range(num_epochs):

    beta = beta_max * min(
        epoch / warmup_epochs,
        1.0
    )

    model.train()

    for hsi in train_loader:

        hsi = hsi.to(device)

        recon, mu, logvar, z = model(hsi)

        l1 = F.l1_loss(recon, hsi)

        mrae = mrae_loss(recon, hsi)

        sam = sam_loss(recon, hsi)

        kl = kl_loss(mu, logvar)

        recon_loss = (
            0.5 * l1
            + 0.4 * mrae
            + 0.1 * sam
        )

        loss = recon_loss + beta * kl

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch}",
        f"Recon={recon_loss.item():.4f}",
        f"KL={kl.item():.4f}",
        f"Beta={beta:.2e}"
    )



#To save decoder
torch.save(
    model.decoder.state_dict(),
    "hsi_decoder.pth"
)
