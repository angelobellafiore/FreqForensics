import torch
import torch.nn as nn


class CrossBranchAttentionFusion(nn.Module):
    """Three-way cross-attention fusion over spatial, LF, and HF feature vectors.

    Each branch attends to each other branch, producing input-dependent residual
    updates. The three updated vectors are concatenated into a 2304-d fused
    representation for the classifier head.

    Input:  F_s  (B, 1792), F_lf (B, 256), F_hf (B, 256)
    Output: (B, 2304) , concat of the three attended feature vectors

    Attention mechanism for a (query Q, key-value KV) pair:
      1. Project Q and KV to a common d_k=128 space
      2. Scalar attention weight: a = sigmoid(dot(Q_proj, K_proj))
      3. Value update: V = Linear(dim_KV -> dim_KV)(KV)
      4. Residual: Q' = Q + a * V
    """

    D_K = 128   # common projection dimension for attention dot-product

    def __init__(self) -> None:
        super().__init__()

        # --- S <-> LF attention ---
        self.q_s_lf  = nn.Linear(1792, self.D_K)   # S queries LF
        self.k_lf_s  = nn.Linear(256,  self.D_K)
        self.v_lf_s  = nn.Linear(256,  1792)   # value must match Q's dim (S=1792)

        self.q_lf_s  = nn.Linear(256,  self.D_K)   # LF queries S
        self.k_s_lf  = nn.Linear(1792, self.D_K)
        self.v_s_lf  = nn.Linear(1792, 256)    # value must match Q's dim (LF=256)

        # --- S <-> HF attention ---
        self.q_s_hf  = nn.Linear(1792, self.D_K)   # S queries HF
        self.k_hf_s  = nn.Linear(256,  self.D_K)
        self.v_hf_s  = nn.Linear(256,  1792)   # value must match Q's dim (S=1792)

        self.q_hf_s  = nn.Linear(256,  self.D_K)   # HF queries S
        self.k_s_hf  = nn.Linear(1792, self.D_K)
        self.v_s_hf  = nn.Linear(1792, 256)    # value must match Q's dim (HF=256)

        # --- LF <-> HF attention ---
        self.q_lf_hf = nn.Linear(256,  self.D_K)   # LF queries HF
        self.k_hf_lf = nn.Linear(256,  self.D_K)
        self.v_hf_lf = nn.Linear(256,  256)

        self.q_hf_lf = nn.Linear(256,  self.D_K)   # HF queries LF
        self.k_lf_hf = nn.Linear(256,  self.D_K)
        self.v_lf_hf = nn.Linear(256,  256)

    @staticmethod
    def _attend(
        q: torch.Tensor,
        q_proj: nn.Linear,
        k: torch.Tensor,
        k_proj: nn.Linear,
        v: torch.Tensor,
        v_proj: nn.Linear,
    ) -> torch.Tensor:
        """Apply one directional attention: Q attends to KV.

        Returns the residual-updated Q' = Q + sigmoid(dot(Q_proj, K_proj)) * V_proj(KV).
        The dot-product is summed over the D_K dimension to produce one scalar
        per sample, a degenerate single-token attention weight.
        """
        a = torch.sigmoid((q_proj(q) * k_proj(k)).sum(dim=1, keepdim=True))  # (B, 1)
        return q + a * v_proj(v)

    def forward(
        self,
        f_s:  torch.Tensor,   # (B, 1792)
        f_lf: torch.Tensor,   # (B, 256)
        f_hf: torch.Tensor,   # (B, 256)
    ) -> torch.Tensor:
        """
        Returns:
            (B, 2304) fused feature vector, concat of attended F_s', F_lf', F_hf'
        """
        # S <-> LF
        f_s  = self._attend(f_s,  self.q_s_lf,  f_lf, self.k_lf_s,  f_lf, self.v_lf_s)
        f_lf = self._attend(f_lf, self.q_lf_s,  f_s,  self.k_s_lf,  f_s,  self.v_s_lf)

        # S <-> HF
        f_s  = self._attend(f_s,  self.q_s_hf,  f_hf, self.k_hf_s,  f_hf, self.v_hf_s)
        f_hf = self._attend(f_hf, self.q_hf_s,  f_s,  self.k_s_hf,  f_s,  self.v_s_hf)

        # LF <-> HF
        f_lf = self._attend(f_lf, self.q_lf_hf, f_hf, self.k_hf_lf, f_hf, self.v_hf_lf)
        f_hf = self._attend(f_hf, self.q_hf_lf, f_lf, self.k_lf_hf, f_lf, self.v_lf_hf)

        return torch.cat([f_s, f_lf, f_hf], dim=1)   # (B, 1792+256+256) = (B, 2304)


if __name__ == '__main__':
    print("Running smoke test for fusion.py...")

    model = CrossBranchAttentionFusion()
    model.eval()

    B = 4
    f_s  = torch.rand(B, 1792)
    f_lf = torch.rand(B, 256)
    f_hf = torch.rand(B, 256)

    with torch.no_grad():
        out = model(f_s, f_lf, f_hf)

    assert out.shape == (B, 2304), f"Expected ({B}, 2304), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print(f"  CrossBranchAttentionFusion: f_s{tuple(f_s.shape)}, "
          f"f_lf{tuple(f_lf.shape)}, f_hf{tuple(f_hf.shape)} -> {tuple(out.shape)}  OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\nAll assertions passed.")
