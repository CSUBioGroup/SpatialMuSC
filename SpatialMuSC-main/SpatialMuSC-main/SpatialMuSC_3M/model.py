import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# -----------------------
# Helper for compatible epsilon sampling
# -----------------------
def _randn_like(x: torch.Tensor, generator: torch.Generator = None):
    """Generate N(0,1) noise with optional generator (compatible with older torch)."""
    if generator is not None:
        return torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
    else:
        return torch.randn_like(x)


# =========================
# Core model (GCN + Transformer main branches with DDPM refine on Transformer outputs)
# =========================
class Encoder_overall(Module):
    """
    Overall encoder with parallel local (GCN) and global (Transformer) branches per modality.
    Transformer outputs are refined by a spatial-graph-conditional DDPM (only applied to Transformer branch).
    Pipeline per modality m:
      local_m   : GCN on spatial & feature graphs -> concat -> MLP
      globalRaw : Global Transformer over X (all nodes as tokens)
      globalRef : DDPMRefiner.refine(globalRaw | conditioned on spatial adjacency)
      h_m       : Intra-modality fusion [local_m || globalRef] -> MLP
    Cross-modality fusion:
      [h_1 || h_2 || h_3] -> MLP -> h_fused -> decoders (spatial adj) -> recon losses
    """
    def __init__(
        self,
        dim_in_feat_omics1: int,
        dim_out_feat_omics1: int,
        dim_in_feat_omics2: int,
        dim_out_feat_omics2: int,
        dim_in_feat_omics3: int,
        dim_out_feat_omics3: int,
        dropout: float = 0.0,
        act=F.relu,

        enable_local: bool = True,
        enable_global: bool = True,

        # Transformer configs (borrowed-style blocks)
        tfm_hidden: int = 128,
        tfm_layers: int = 2,
        tfm_heads: int = 8,
        tfm_dropout: float = 0.1,

        # DDPM refiner (only for Transformer outputs)
        enable_ddpm: bool = True,
        ddpm_T: int = 1000,
        ddpm_steps_infer: int = 1,   # DDIM-like reverse steps for refinement in forward
        ddpm_beta_start: float = 1e-4,
        ddpm_beta_end: float = 2e-2,
    ):
        super(Encoder_overall, self).__init__()
        # dims
        self.in_dim_1 = dim_in_feat_omics1
        self.in_dim_2 = dim_in_feat_omics2
        self.in_dim_3 = dim_in_feat_omics3
        self.latent_dim_1 = dim_out_feat_omics1
        self.latent_dim_2 = dim_out_feat_omics2
        self.latent_dim_3 = dim_out_feat_omics3
        self.dropout = dropout
        self.act = act

        self.enable_local = enable_local
        self.enable_global = enable_global

        if not self.enable_global:
            enable_ddpm = False

        # ----- Local (GCN) branch -----
        self.enc1 = Encoder(self.in_dim_1, self.latent_dim_1)
        self.enc2 = Encoder(self.in_dim_2, self.latent_dim_2)
        self.enc3 = Encoder(self.in_dim_3, self.latent_dim_3)
        self.dec1 = Decoder(self.latent_dim_1, self.in_dim_1)
        self.dec2 = Decoder(self.latent_dim_2, self.in_dim_2)
        self.dec3 = Decoder(self.latent_dim_3, self.in_dim_3)
        self.intra_local_1 = MLP(self.latent_dim_1 * 2, self.latent_dim_1, self.latent_dim_1)
        self.intra_local_2 = MLP(self.latent_dim_2 * 2, self.latent_dim_2, self.latent_dim_2)
        self.intra_local_3 = MLP(self.latent_dim_3 * 2, self.latent_dim_3, self.latent_dim_3)

        # ----- Global (Transformer) branch -----
        self.tfm1 = GlobalTransformerBorrowed(
            input_dim=self.in_dim_1, hidden_dim=tfm_hidden, out_dim=self.latent_dim_1,
            n_layers=tfm_layers, n_heads=tfm_heads, dropout=tfm_dropout
        )
        self.tfm2 = GlobalTransformerBorrowed(
            input_dim=self.in_dim_2, hidden_dim=tfm_hidden, out_dim=self.latent_dim_2,
            n_layers=tfm_layers, n_heads=tfm_heads, dropout=tfm_dropout
        )
        self.tfm3 = GlobalTransformerBorrowed(
            input_dim=self.in_dim_3, hidden_dim=tfm_hidden, out_dim=self.latent_dim_3,
            n_layers=tfm_layers, n_heads=tfm_heads, dropout=tfm_dropout
        )

        # ----- DDPM refiners (only applied to Transformer outputs) -----
        self.enable_ddpm = enable_ddpm
        if self.enable_ddpm:
            self.refiner1 = DDPMRefiner(
                embed_dim=self.latent_dim_1, T=ddpm_T,
                beta_start=ddpm_beta_start, beta_end=ddpm_beta_end
            )
            self.refiner2 = DDPMRefiner(
                embed_dim=self.latent_dim_2, T=ddpm_T,
                beta_start=ddpm_beta_start, beta_end=ddpm_beta_end
            )
            self.refiner3 = DDPMRefiner(
                embed_dim=self.latent_dim_3, T=ddpm_T,
                beta_start=ddpm_beta_start, beta_end=ddpm_beta_end
            )
        self.ddpm_steps_infer = ddpm_steps_infer

        # ----- Intra-modality fusion (local + refined global) -----
        self.intra_fuse_1 = MLP(self.latent_dim_1 * 2, self.latent_dim_1, self.latent_dim_1)
        self.intra_fuse_2 = MLP(self.latent_dim_2 * 2, self.latent_dim_2, self.latent_dim_2)
        self.intra_fuse_3 = MLP(self.latent_dim_3 * 2, self.latent_dim_3, self.latent_dim_3)

        # ----- Cross-modality fusion -----
        self.fusion = MLP(self.latent_dim_1 + self.latent_dim_2 + self.latent_dim_3, self.latent_dim_1, self.latent_dim_1)

    def forward(
        self,
        x1, x2, x3,                     # [N,F1], [N,F2], [N,F3]
        A1_spatial, A1_feature,         # sparse adj (omics1)
        A2_spatial, A2_feature,         # sparse adj (omics2)
        A3_spatial, A3_feature,         # sparse adj (omics3)
        return_raw_global: bool = False,  # if True, also return raw transformer outputs (for DDPM loss)
    ):
        # densify once and reuse
        A1_spatial_dense = A1_spatial.to_dense() if A1_spatial.is_sparse else A1_spatial
        A1_feature_dense = A1_feature.to_dense() if A1_feature.is_sparse else A1_feature
        A2_spatial_dense = A2_spatial.to_dense() if A2_spatial.is_sparse else A2_spatial
        A2_feature_dense = A2_feature.to_dense() if A2_feature.is_sparse else A2_feature
        A3_spatial_dense = A3_spatial.to_dense() if A3_spatial.is_sparse else A3_spatial
        A3_feature_dense = A3_feature.to_dense() if A3_feature.is_sparse else A3_feature

        N = x1.size(0)

        # ===== Local (GCN) per modality =====
        if self.enable_local:
            z1_sp = self.enc1(x1, A1_spatial_dense)
            z1_ft = self.enc1(x1, A1_feature_dense)
            local1 = self.intra_local_1(torch.cat([z1_sp, z1_ft], dim=1))   # [N, d1]

            z2_sp = self.enc2(x2, A2_spatial_dense)
            z2_ft = self.enc2(x2, A2_feature_dense)
            local2 = self.intra_local_2(torch.cat([z2_sp, z2_ft], dim=1))   # [N, d2]

            z3_sp = self.enc3(x3, A3_spatial_dense)
            z3_ft = self.enc3(x3, A3_feature_dense)
            local3 = self.intra_local_3(torch.cat([z3_sp, z3_ft], dim=1))   # [N, d3]
        else:
            local1 = torch.zeros(N, self.latent_dim_1, device=x1.device, dtype=x1.dtype)
            local2 = torch.zeros(N, self.latent_dim_2, device=x2.device, dtype=x2.dtype)
            local3 = torch.zeros(N, self.latent_dim_3, device=x3.device, dtype=x3.dtype)

        # ===== Global (Transformer) per modality =====
        if self.enable_global:
            global_raw_1 = self.tfm1(x1)  # [N, d1]
            global_raw_2 = self.tfm2(x2)  # [N, d2]
            global_raw_3 = self.tfm3(x3)  # [N, d3]
        else:
            global_raw_1 = torch.zeros(N, self.latent_dim_1, device=x1.device, dtype=x1.dtype)
            global_raw_2 = torch.zeros(N, self.latent_dim_2, device=x2.device, dtype=x2.dtype)
            global_raw_3 = torch.zeros(N, self.latent_dim_3, device=x3.device, dtype=x3.dtype)

        # ===== DDPM refinement (only on Transformer outputs) =====
        if self.enable_ddpm:
            global_ref_1 = self.refiner1.refine(global_raw_1, A1_spatial, steps=self.ddpm_steps_infer)
            global_ref_2 = self.refiner2.refine(global_raw_2, A2_spatial, steps=self.ddpm_steps_infer)
            global_ref_3 = self.refiner3.refine(global_raw_3, A3_spatial, steps=self.ddpm_steps_infer)
        else:
            global_ref_1, global_ref_2, global_ref_3 = global_raw_1, global_raw_2, global_raw_3

        # ===== Intra-modality fusion (local + global_refined) =====
        h1 = self.intra_fuse_1(torch.cat([local1, global_ref_1], dim=1))
        h2 = self.intra_fuse_2(torch.cat([local2, global_ref_2], dim=1))
        h3 = self.intra_fuse_3(torch.cat([local3, global_ref_3], dim=1))

        # ===== Cross-modality fusion =====
        h_fused = self.fusion(torch.cat([h1, h2, h3], dim=1))  # [N, d1]

        # ===== Reconstruction (use spatial adj) =====
        xrec1 = self.dec1(h_fused, A1_spatial_dense)
        xrec2 = self.dec2(h_fused, A2_spatial_dense)
        xrec3 = self.dec3(h_fused, A3_spatial_dense)

        out = {
            "emb_local_omics1": local1,
            "emb_local_omics2": local2,
            "emb_local_omics3": local3,
            "emb_global_omics1": global_ref_1,      # refined global for contrast / fusion
            "emb_global_omics2": global_ref_2,
            "emb_global_omics3": global_ref_3,
            "emb_latent_omics1": h1,
            "emb_latent_omics2": h2,
            "emb_latent_omics3": h3,
            "emb_latent_combined": h_fused,
            "emb_recon_omics1": xrec1,
            "emb_recon_omics2": xrec2,
            "emb_recon_omics3": xrec3,
        }
        if return_raw_global:
            out["emb_global_raw_omics1"] = global_raw_1
            out["emb_global_raw_omics2"] = global_raw_2
            out["emb_global_raw_omics3"] = global_raw_3
        return out


# =========================
# GCN encoder / decoder / MLP
# =========================
class Encoder(Module):
    """ Modality-specific GCN encoder (linear projection + graph propagation). """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        h = torch.mm(x, self.weight)
        # deterministic graph matmul (avoid sparse CUDA non-determinism)
        if adj.is_sparse:
            h = torch.mm(adj.to_dense(), h)
        else:
            h = torch.mm(adj, h)
        return h


class Decoder(Module):
    """ Modality-specific GCN decoder (linear projection + graph propagation). """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, z, adj):
        x = torch.mm(z, self.weight)
        # deterministic graph matmul
        if adj.is_sparse:
            x = torch.spmm(adj, x)
        else:
            x = torch.mm(adj, x)
        return x


class MLP(nn.Module):
    """ Two-layer MLP used for aggregation and fusion. """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float = 0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            torch.nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            torch.nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


# =========================
# Global Transformer (borrowed-style blocks)
# =========================
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GlobalTransformerBorrowed(nn.Module):
    """
    Borrowed-style Transformer encoder over node set (nodes as tokens).
    X [N, F] -> in_proj -> [1, N, H] -> EncoderLayer*L -> LN -> out_proj -> [N, D]
    """
    def __init__(self, input_dim, hidden_dim, out_dim, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoders = [
            EncoderLayer(hidden_dim, 2 * hidden_dim, dropout, 0.1, n_heads)
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.apply(lambda m: init_params(m, n_layers=n_layers))

    def forward(self, x):  # x: [N, F]
        x = x.unsqueeze(0)              # [1, N, F]
        h = self.input_proj(x)          # [1, N, H]
        for layer in self.layers:
            h = layer(h)
        h = self.final_ln(h)            # [1, N, H]
        out = self.out_proj(h).squeeze(0)  # [N, D]
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        B = q.size(0)

        q = self.linear_q(q).view(B, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(B, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(B, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)               # [B, Hh, Lq, d_k]
        v = v.transpose(1, 2)               # [B, Hh, Lv, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [B, Hh, d_k, Lk]

        q = q * self.scale
        att = torch.matmul(q, k)            # [B, Hh, Lq, Lk]
        if attn_bias is not None:
            att = att + attn_bias
        att = torch.softmax(att, dim=-1)
        att = self.att_dropout(att)

        out = att.matmul(v)                 # [B, Hh, Lq, d_v]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * d_v)  # [B, Lq, H]
        out = self.output_layer(out)
        assert out.size() == orig_q_size
        return out


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        def _forward(x, attn_bias=None):
            y = self.self_attention_norm(x)
            y = self.self_attention(y, y, y, attn_bias)
            y = self.self_attention_dropout(y)
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
            return x
        self._forward_impl = _forward

    def forward(self, x, attn_bias=None):
        return self._forward_impl(x, attn_bias)


# =========================
# Contrast head (borrowed)
# =========================
class Contrast(nn.Module):
    """
    Symmetric InfoNCE-style contrast between two views of same nodes.
    Borrowed structure: projection -> exp(cos/τ) -> row-softmax -> -log P_ii averaged (both directions).
    """
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos_dense):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log((matrix_mp2sc * pos_dense).sum(dim=-1) + 1e-12).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log((matrix_sc2mp * pos_dense).sum(dim=-1) + 1e-12).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc


# =========================
# DDPM refiner (only applied to Transformer outputs)
# =========================
def _time_sinusoidal_embedding(t, dim, device):
    """ sinusoidal time embedding for diffusion step t (shape [B, dim]) """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1 + 1e-8))
    )
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class GraphEpsNet(nn.Module):
    def __init__(self, embed_dim: int, time_dim: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_t: torch.Tensor, t_emb: torch.Tensor, A_norm: torch.Tensor):
        h = F.gelu(self.lin1(x_t))
        if A_norm.is_sparse:
            h = torch.mm(A_norm.to_dense(), h)
        else:
            h = torch.mm(A_norm, h)
        h = self.lin2(h)
        te = self.time_mlp(t_emb)
        h = h + te
        h = self.norm(h)
        return h


class DDPMRefiner(nn.Module):
    def __init__(self, embed_dim: int, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.embed_dim = embed_dim
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.eps_net = GraphEpsNet(embed_dim=embed_dim, time_dim=128)
        self.gen = None

    @staticmethod
    def _normalize_adj(A: torch.Tensor, eps: float = 1e-12, add_self_loop: bool = True):
        if add_self_loop:
            N = A.size(0)
            I_idx = torch.arange(N, device=A.device).repeat(2, 1)
            I = torch.sparse_coo_tensor(I_idx, torch.ones(N, device=A.device), (N, N))
            A = A + I
        deg = torch.sparse.sum(A, dim=1).to_dense() + eps
        d_inv_sqrt = deg.pow(-0.5)
        return A, d_inv_sqrt

    def diffusion_loss(self, z0: torch.Tensor, A: torch.Tensor):
        N, D = z0.shape
        device = z0.device
        g = self.gen

        t = torch.randint(1, self.T + 1, (N,), device=device, generator=g)
        sqrt_ab = self.sqrt_alphas_cumprod[t - 1].unsqueeze(1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t - 1].unsqueeze(1)

        eps = _randn_like(z0, generator=g)
        x_t = sqrt_ab * z0 + sqrt_omab * eps

        t_emb = _time_sinusoidal_embedding(t, dim=128, device=device)

        A_hat, d_inv_sqrt = self._normalize_adj(A)
        A_hat_dense = A_hat.to_dense() if A_hat.is_sparse else A_hat
        d_col = d_inv_sqrt.unsqueeze(1)
        x_t_hat = d_col * (A_hat_dense @ (d_col * x_t))

        eps_pred = self.eps_net(x_t_hat, t_emb, A_hat_dense)
        loss = F.mse_loss(eps_pred, eps)
        return loss

    def refine(self, z0: torch.Tensor, A: torch.Tensor, steps: int = 1):
        if steps <= 0:
            return z0

        N = z0.size(0)
        device = z0.device
        ts = torch.linspace(self.T, 1, steps, device=device).long()

        x_t = z0.clone()
        A_hat, d_inv_sqrt = self._normalize_adj(A)
        A_hat_dense = A_hat.to_dense() if A_hat.is_sparse else A_hat
        d_col = d_inv_sqrt.unsqueeze(1)

        for t in ts:
            t_batch = torch.full((N,), int(t.item()), device=device, dtype=torch.long)
            t_emb = _time_sinusoidal_embedding(t_batch, dim=128, device=device)

            x_hat = d_col * (A_hat_dense @ (d_col * x_t))
            eps_pred = self.eps_net(x_hat, t_emb, A_hat_dense)

            s_ab = self.sqrt_alphas_cumprod[t - 1]
            s_omab = self.sqrt_one_minus_alphas_cumprod[t - 1]
            sqrt_ab = s_ab.view(1, 1).expand(N, 1)
            sqrt_omab = s_omab.view(1, 1).expand(N, 1)

            x0_hat = (x_t - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)

            if t.item() > 1:
                s_ab_prev = self.sqrt_alphas_cumprod[t - 2]
                sqrt_ab_prev = s_ab_prev.view(1, 1).expand(N, 1)
                x_t = sqrt_ab_prev * x0_hat
            else:
                x_t = x0_hat

        return x_t
