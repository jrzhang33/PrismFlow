import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm

from .backbone import TransformerBackbone


class PrismFlow(nn.Module):
    def __init__(
        self,
        seq_length,
        feature_size,
        n_layer_enc=3,
        n_layer_dec=6,
        d_model=None,
        n_heads=4,
        mlp_hidden_times=4,
        attn_pd=0.0,
        resid_pd=0.0,
        timesteps=1000,
        sampling_timesteps=None,
        prediction='velocity',
        hybrid_switch_time=0.97,
        hybrid_late_fraction=0.2,
        n_experts=4,
        dz=32,
        dissipation_rank=None,
        delta_init=0.01,
        delta_learnable=True,
        delta_min=0.0,
        time_emb_dim=None,
        lambda_wta=0.0,
        lambda_bal=0.0,
        wta_beta=0.1,
        wta_eps=1e-8,
        lambda_power=2.0,
        warmup_ratio=0.2,
        max_epochs=10000,
        residual_scale=0.0,
        noise_scale=1.0,
    ):
        super().__init__()
        assert prediction in ('velocity', 'hybrid')
        assert lambda_power > 0

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps
        self.prediction = prediction
        self.hybrid_switch_time = hybrid_switch_time
        self.hybrid_late_fraction = hybrid_late_fraction

        self.n_modes = n_experts
        self.dz = dz
        self.dissipation_rank = dissipation_rank if dissipation_rank is not None else dz
        self.delta_learnable = delta_learnable
        self.delta_min = delta_min

        self.lambda_wta = lambda_wta
        self.lambda_bal = lambda_bal
        self.wta_beta = wta_beta
        self.wta_eps = wta_eps
        self.lambda_power = lambda_power
        self.warmup_ratio = warmup_ratio
        self.max_epochs = max_epochs

        self.residual_scale = residual_scale
        self.noise_scale = noise_scale

        dim_model = d_model if d_model is not None else 4 * n_heads * mlp_hidden_times
        self.time_emb_dim = time_emb_dim if time_emb_dim is not None else dim_model // 4

        self.model = TransformerBackbone(
            n_feat=feature_size,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_embd=dim_model,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
        )

        self.global_query = nn.Parameter(torch.randn(1, 1, dim_model))
        self.global_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=n_heads, dropout=attn_pd,
                                                 batch_first=True)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.ReLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        self.mode_encoder = nn.Sequential(
            nn.Linear(dim_model + self.time_emb_dim, dim_model // 2),
            nn.ReLU(),
            nn.Dropout(resid_pd),
            nn.Linear(dim_model // 2, n_experts),
        )

        self.projector = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.ReLU(),
            nn.Dropout(resid_pd),
            nn.Linear(dim_model // 2, dz),
        )

        self.dynamics_S = nn.Parameter(torch.randn(n_experts, dz, dz) * 0.01)
        self.dynamics_R = nn.Parameter(torch.randn(n_experts, self.dissipation_rank, dz) * 0.01)
        if delta_learnable:
            delta_raw_init = torch.log(torch.exp(torch.tensor(delta_init - delta_min)) - 1.0)
            self.delta_raw = nn.Parameter(delta_raw_init.repeat(n_experts))
        else:
            self.register_buffer('delta_raw', torch.tensor([delta_init] * n_experts))

        self.dynamics_decoder = nn.Sequential(
            nn.Linear(2 * dz, dim_model),
            nn.ReLU(),
            nn.Dropout(resid_pd),
            nn.Linear(dim_model, seq_length * feature_size),
        )

        self.register_buffer('betas', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('_training_step', torch.tensor(0, dtype=torch.long))

    def get_dissipation_strengths(self):
        if self.delta_learnable:
            return self.delta_min + F.softplus(self.delta_raw)
        return self.delta_raw

    def get_dynamics_matrices(self):
        delta = self.get_dissipation_strengths()
        eye = torch.eye(self.dz, device=self.dynamics_S.device)
        S, R = self.dynamics_S, self.dynamics_R
        return (S - S.transpose(1, 2)) - torch.matmul(R.transpose(1, 2), R) - delta[:, None, None] * eye

    def discretize(self, t):
        if t.dim() > 1:
            t = t.view(-1)
        return (torch.clamp(t, 0.0, 1.0) * (self.num_timesteps - 1)).long()

    def time_weight(self, t):
        return torch.clamp(t ** self.lambda_power, max=0.95)

    def to_velocity(self, out, x, t):
        if self.prediction == 'velocity':
            return out
        v_from_x1 = (out - x) / torch.clamp(1.0 - t, min=1e-2)[:, None, None]
        return torch.where((t >= self.hybrid_switch_time)[:, None, None], out, v_from_x1)

    def global_velocity(self, x, t):
        out, latent = self.model(x, self.discretize(t), return_enc=True)
        return self.to_velocity(out, x, t), latent

    def route(self, latent, t):
        h, _ = self.global_attn(self.global_query.expand(latent.shape[0], -1, -1), latent, latent)
        h = h.squeeze(1)
        logits = self.mode_encoder(torch.cat([h, self.time_embedding(t.unsqueeze(-1))], dim=-1))
        return h, logits

    def expert_velocities(self, h):
        z = self.projector(h)
        Az = torch.einsum('kij,bj->bki', self.get_dynamics_matrices(), z)
        r = self.dynamics_decoder(torch.cat([z[:, None].expand_as(Az), Az], dim=-1))
        return r.view(h.shape[0], self.n_modes, self.seq_length, self.feature_size)

    def wta_loss(self, x_t, x_start, t, v_global, latent):
        h, logits = self.route(latent, t)
        q = F.softmax(logits, dim=-1)
        r = self.expert_velocities(h)

        u_tilde = v_global.detach()[:, None] + r
        x1_hat = x_t[:, None] + (1.0 - t)[:, None, None, None] * u_tilde
        errors = reduce((x1_hat - x_start[:, None]) ** 2, 'b k ... -> b k', 'mean')
        scores = errors - self.wta_beta * torch.log(q + self.wta_eps)
        wta = scores.min(dim=1)[0]

        q_mean = q.mean(dim=0).clamp_min(1e-8)
        bal = (torch.log(torch.full_like(q_mean, 1.0 / self.n_modes)) - torch.log(q_mean)).mean()
        return wta, bal

    def forward(self, x_start):
        noise = torch.randn_like(x_start)
        t = torch.rand(x_start.shape[0], device=x_start.device)
        if self.prediction == 'hybrid':
            ts = self.hybrid_switch_time
            late = torch.rand_like(t) < self.hybrid_late_fraction
            t = torch.where(late, ts + (1.0 - ts) * t, ts * t)
        x_t = (1.0 - t)[:, None, None] * noise + t[:, None, None] * x_start
        pred, latent = self.model(x_t, self.discretize(t), return_enc=True)
        if self.prediction == 'hybrid':
            late = (t >= self.hybrid_switch_time)[:, None, None]
            target = torch.where(late, x_start - noise, x_start)
        else:
            target = x_start - noise
        loss_cfm = reduce(F.mse_loss(pred, target, reduction='none'), 'b ... -> b', 'mean').mean()

        if self.lambda_wta <= 0 and self.lambda_bal <= 0:
            return loss_cfm

        warmup_steps = int(self.warmup_ratio * self.max_epochs)
        warmup = min(1.0, self._training_step.item() / max(1, warmup_steps))

        wta, bal = self.wta_loss(x_t, x_start, t, self.to_velocity(pred, x_t, t), latent)
        loss_wta = warmup * (self.time_weight(t) * wta).mean()
        loss_bal = warmup * bal
        return loss_cfm + self.lambda_wta * loss_wta + self.lambda_bal * loss_bal

    def velocity(self, x, t):
        v, latent = self.global_velocity(x, t)
        if self.residual_scale > 0:
            h, logits = self.route(latent, t)
            k = torch.argmax(logits, dim=-1)
            r = self.expert_velocities(h)[torch.arange(x.shape[0], device=x.device), k]
            v = v + self.residual_scale * self.time_weight(t)[:, None, None] * r
        return v

    def integrate(self, x, steps):
        dt = 1.0 / float(steps)
        for i in tqdm(range(steps), desc='PrismFlow sampling', leave=False):
            t = torch.full((x.shape[0],), i / float(steps), device=x.device)
            v = self.velocity(x, t)
            x = x + dt * v
        return x

    @torch.no_grad()
    def generate(self, batch_size):
        x0 = torch.randn((batch_size, self.seq_length, self.feature_size), device=self.betas.device)
        if self.noise_scale != 1.0:
            x0 = x0 * self.noise_scale
        return self.integrate(x0, self.sampling_timesteps)
