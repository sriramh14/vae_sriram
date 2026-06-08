class HSIBetaVAE(nn.Module):
    def __init__(
        self,
        in_channels=31,
        latent_channels=8
    ):
        super().__init__()

        self.encoder = HSIEncoder(
            in_channels,
            latent_channels
        )

        self.reparam = Reparameterization()

        self.decoder = HSIDecoder(
            latent_channels,
            in_channels
        )

    def forward(self, x):

        mu, logvar = self.encoder(x)

        z = self.reparam(mu, logvar)

        recon = self.decoder(z)

        return recon, mu, logvar, z
