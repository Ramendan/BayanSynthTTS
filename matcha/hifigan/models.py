import torch


class MultiPeriodDiscriminator:
    """Lightweight stub of Matcha's MultiPeriodDiscriminator used only so
    the YAML/config parser and HiFiGAN discriminator construction succeed in
    dry-run / UI mode. This is NOT a real MPD implementation.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        # Return placeholders consistent with HiFiGan discriminator API
        # (y_d_rs, y_d_gs, fmap_rs, fmap_gs)
        return None, None, [], []


# Minimal loss utilities expected by cosyvoice.hifigan.hifigan
def feature_loss(fmap_rs, fmap_gs):
    import torch
    return torch.zeros(1)


def generator_loss(y_d_gs):
    import torch
    # return (loss_gen, extra_info)
    return torch.zeros(1), {}


def discriminator_loss(y_d_rs, y_d_gs):
    import torch
    # return (loss_disc, extra1, extra2)
    return torch.zeros(1), None, None
