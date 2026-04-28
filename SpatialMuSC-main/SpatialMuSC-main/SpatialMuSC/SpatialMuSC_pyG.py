import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .model import Encoder_overall, Contrast
from .preprocess import adjacent_matrix_preprocessing

class FusionInfoNCELoss(nn.Module):
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau
    def forward(self, z_full: torch.Tensor, z_masked: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z_full, dim=-1)
        z2 = F.normalize(z_masked, dim=-1)
        N = z1.size(0)

        sim_12 = torch.mm(z1, z2.t()) / self.tau
        labels = torch.arange(N, device=z1.device)
        loss_12 = F.cross_entropy(sim_12, labels)

        sim_21 = torch.mm(z2, z1.t()) / self.tau
        loss_21 = F.cross_entropy(sim_21, labels)

        loss = 0.5 * (loss_12 + loss_21)
        return loss

class Train_SpatialMuSC:
    def __init__(self,
                 data,
                 datatype='SPOTS',
                 device=torch.device('cpu'),
                 random_seed=2022,
                 learning_rate=0.0001,
                 weight_decay=0.00,
                 epochs=800,
                 dim_output=64,
                 weight_factors=[1, 1],
                 # Transformer / DDPM / Contrast
                 tfm_hidden=128, tfm_layers=2, tfm_heads=8, tfm_dropout=0.0,
                 enable_ddpm=True, ddpm_T=500, ddpm_steps_infer=1,
                 ddpm_beta_start=1e-4, ddpm_beta_end=2e-2,
                 ddpm_lambda=0.2,             # loss weight for diffusion (per modality)
                 enable_contrast=True, contrast_tau=0.1, lambda_lg=0.1,
                 enable_infmask=True,
                 enable_local=True,
                 enable_global=True):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        self.random_seed = random_seed

        # adj & features
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.A1_sp = self.adj['adj_spatial_omics1'].to(self.device)
        self.A2_sp = self.adj['adj_spatial_omics2'].to(self.device)
        self.A1_ft = self.adj['adj_feature_omics1'].to(self.device)
        self.A2_ft = self.adj['adj_feature_omics2'].to(self.device)

        self.X1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.X2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.in_dim_1 = self.X1.shape[1]
        self.in_dim_2 = self.X2.shape[1]
        self.out_dim_1 = self.dim_output
        self.out_dim_2 = self.dim_output


        if self.datatype == 'SPOTS':
            self.epochs = 1000
            self.weight_factors = [1, 10]
            self.lambda_infmask = 0.8
            lambda_lg = 0.2
        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 800
            self.weight_factors = [1, 50]
            tfm_heads = 4
            self.lambda_infmask = 0.1
            lambda_lg = 0.05
        elif self.datatype == '10x':
            self.epochs = 800
            self.lambda_infmask = 0.01
        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs = 1100
            self.weight_factors = [1, 5]
            self.lambda_infmask = 0.6
            lambda_lg = 0.2
            ddpm_lambda = 0.5

        # -------------------------------
        if not enable_global:
            enable_ddpm = False
        if (not enable_local) or (not enable_global):
            enable_contrast = False
        # -------------------------------

        # model
        self.model = Encoder_overall(
            self.in_dim_1, self.out_dim_1, self.in_dim_2, self.out_dim_2,
            tfm_hidden=tfm_hidden, tfm_layers=tfm_layers, tfm_heads=tfm_heads, tfm_dropout=tfm_dropout,
            enable_ddpm=enable_ddpm, ddpm_T=ddpm_T, ddpm_steps_infer=ddpm_steps_infer,
            ddpm_beta_start=ddpm_beta_start, ddpm_beta_end=ddpm_beta_end,
            enable_local=enable_local,
            enable_global=enable_global
        ).to(self.device)

        # contrast
        self.enable_contrast = enable_contrast
        self.lambda_lg = lambda_lg
        if self.enable_contrast:
            self.contrast1 = Contrast(hidden_dim=self.out_dim_1, tau=contrast_tau, lam=0.5).to(self.device)
            self.contrast2 = Contrast(hidden_dim=self.out_dim_2, tau=contrast_tau, lam=0.5).to(self.device)

        # ddpm loss weight
        self.enable_ddpm = enable_ddpm
        self.ddpm_lambda = ddpm_lambda

        # dedicated RNG for diffusion
        if self.enable_ddpm:
            self._ddpm_gen = torch.Generator(device=self.device)
            self._ddpm_gen.manual_seed(self.random_seed)
            self.model.refiner1.gen = self._ddpm_gen
            self.model.refiner2.gen = self._ddpm_gen

        self.enable_infmask = enable_infmask
        if self.enable_infmask:
            self.fusion_infmask = FusionInfoNCELoss(tau=0.1).to(self.device)

    def _build_spatial_pos_dense(self, A_sp, N, topk=5, self_weight=0.6, neighbor_weight=0.4):
        A = A_sp.to_dense() if A_sp.is_sparse else A_sp
        A = A.clone()
        idx = torch.arange(N, device=A.device)
        A[idx, idx] = 0.0

        k = min(topk, max(1, N - 1))
        vals, nn_idx = torch.topk(A, k=k, dim=1, largest=True)

        mask = torch.zeros_like(A, dtype=torch.bool)
        mask.scatter_(1, nn_idx, True)
        A = torch.where(mask, A, torch.zeros_like(A))

        row_sum = A.sum(dim=1, keepdim=True) + 1e-12
        neigh_prob = (A / row_sum) * neighbor_weight

        pos = torch.zeros_like(A)
        pos[idx, idx] = self_weight
        pos = pos + neigh_prob

        pos = pos / (pos.sum(dim=1, keepdim=True) + 1e-12)
        return pos

    def _feature_mask(self, X, keep_ratio: float = 0.2):
        N, Fdim = X.shape
        mask = torch.bernoulli(torch.full((Fdim,), keep_ratio, device=X.device))
        if mask.sum() == 0:
            mask[torch.randint(0, Fdim, (1,), device=X.device)] = 1.
        return X * mask.unsqueeze(0)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()

        if self.enable_contrast:
            N = self.X1.shape[0]
            self.pos_dense1 = self._build_spatial_pos_dense(self.A1_sp, N, topk=5, self_weight=0.6, neighbor_weight=0.4).to(self.device)
            self.pos_dense2 = self._build_spatial_pos_dense(self.A2_sp, N, topk=5, self_weight=0.6, neighbor_weight=0.4).to(self.device)

        for epoch in tqdm(range(self.epochs)):
            seed_e = self.random_seed + epoch
            torch.manual_seed(seed_e)
            torch.cuda.manual_seed(seed_e)
            torch.cuda.manual_seed_all(seed_e)

            self.model.train()
            out = self.model(
                self.X1, self.X2,
                self.A1_sp, self.A1_ft,
                self.A2_sp, self.A2_ft,
                return_raw_global=True
            )

            loss_recon1 = F.mse_loss(out['emb_recon_omics1'], self.X1)
            loss_recon2 = F.mse_loss(out['emb_recon_omics2'], self.X2)
            loss = self.weight_factors[0] * loss_recon1 + self.weight_factors[1] * loss_recon2

            if self.enable_contrast:
                lg1 = self.contrast1(out['emb_local_omics1'], out['emb_global_omics1'], self.pos_dense1)
                lg2 = self.contrast2(out['emb_local_omics2'], out['emb_global_omics2'], self.pos_dense2)
                contrast_loss = self.lambda_lg * (lg1 + lg2)
                loss = loss + contrast_loss
            else:
                contrast_loss = torch.tensor(0.0, device=self.device)

            if self.enable_ddpm:
                z0_1 = out['emb_global_raw_omics1']
                z0_2 = out['emb_global_raw_omics2']
                ddpm_loss = self.model.refiner1.diffusion_loss(z0_1, self.A1_sp) + \
                            self.model.refiner2.diffusion_loss(z0_2, self.A2_sp)
                ddpm_loss = self.ddpm_lambda * ddpm_loss
                loss = loss + ddpm_loss
            else:
                ddpm_loss = torch.tensor(0.0, device=self.device)

            if self.enable_infmask:
                z_full = out['emb_latent_combined']

                X1_m = self._feature_mask(self.X1, keep_ratio=0.2)
                X2_m = self._feature_mask(self.X2, keep_ratio=0.2)
                out_mask = self.model(
                    X1_m, X2_m,
                    self.A1_sp, self.A1_ft,
                    self.A2_sp, self.A2_ft,
                    return_raw_global=False
                )
                z_mask = out_mask['emb_latent_combined']

                infmask_loss = self.fusion_infmask(z_full, z_mask)
                loss = loss + self.lambda_infmask * infmask_loss
            else:
                infmask_loss = torch.tensor(0.0, device=self.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            out = self.model(self.X1, self.X2, self.A1_sp, self.A1_ft, self.A2_sp, self.A2_ft)

        emb1 = F.normalize(out['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb2 = F.normalize(out['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_fused = F.normalize(out['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        return {
            'emb_latent_omics1': emb1.detach().cpu().numpy(),
            'emb_latent_omics2': emb2.detach().cpu().numpy(),
            'SpatialMuSC': emb_fused.detach().cpu().numpy()
        }