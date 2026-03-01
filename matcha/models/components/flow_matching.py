import torch


class BASECFM(torch.nn.Module):
    """Very small stub of the real Matcha BASECFM used only to satisfy imports
    and YAML parsing during local dry‑run / UI startup. This does NOT implement
    the real flow-matching functionality — it's a placeholder.
    """

    def __init__(self, n_feats=None, cfm_params=None, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.n_feats = n_feats
        self.cfm_params = cfm_params
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        # Provide sensible defaults referenced by CosyVoice code
        self.t_scheduler = getattr(cfm_params, "t_scheduler", "cosine") if cfm_params is not None else "cosine"
        self.training_cfg_rate = getattr(cfm_params, "training_cfg_rate", 0.0) if cfm_params is not None else 0.0
        self.inference_cfg_rate = getattr(cfm_params, "inference_cfg_rate", 0.0) if cfm_params is not None else 0.0
        self.sigma_min = getattr(cfm_params, "sigma_min", 1e-6) if cfm_params is not None else 1e-6

    def forward(self, *args, **kwargs):
        raise NotImplementedError("BASECFM is a stub placeholder")

    def solve_euler(self, *args, **kwargs):
        raise NotImplementedError("BASECFM is a stub placeholder")
