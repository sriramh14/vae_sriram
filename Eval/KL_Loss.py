def kl_loss(mu, logvar):

    kl = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl.mean()
