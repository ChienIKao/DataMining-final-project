from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

GRID_SIZE = 200
TIME_STEPS = 48


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, latent_dim: int = 64, condition_dim: int = 40):
        super().__init__()
        self.condition_embed = nn.Linear(condition_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        _, (hidden, _) = self.lstm(x)
        traj_embed = hidden[-1]
        cond_embed = torch.relu(self.condition_embed(condition))
        joined = torch.cat([traj_embed, cond_embed], dim=-1)
        return self.fc_mu(joined), self.fc_log_var(joined)


class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, output_dim: int = 2, condition_dim: int = 40):
        super().__init__()
        self.input_embed = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        seed = torch.relu(self.input_embed(torch.cat([z, condition], dim=-1)))
        sequence = seed.unsqueeze(1).expand(-1, TIME_STEPS, -1)
        output, _ = self.lstm(sequence)
        return torch.sigmoid(self.fc_out(output))


class CVAE(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, latent_dim: int = 64, condition_dim: int = 40):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = TrajectoryDecoder(latent_dim, hidden_dim, input_dim, condition_dim)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * log_var)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encoder(x, condition)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z, condition), mu, log_var

    def sample(self, condition: Tensor, n_samples: int = 1) -> Tensor:
        condition = condition.repeat_interleave(n_samples, dim=0)
        z = torch.randn(condition.size(0), self.latent_dim, device=condition.device)
        return self.decoder(z, condition)


def cvae_loss(x_recon: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, beta: float = 1.0) -> Tensor:
    recon = F.mse_loss(x_recon, x, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    return recon + beta * kl


def build_condition_vector(
    uid: int,
    d: int,
    user_hotspots: pd.DataFrame,
    user_stability: pd.DataFrame,
    grid_poi: pd.DataFrame,
    cluster_map: pd.DataFrame,
    n_clusters: int,
) -> np.ndarray:
    """Build a numeric condition vector for one user/day."""
    values: list[float] = []
    hotspot_row = user_hotspots[user_hotspots["uid"].eq(uid)]
    for idx in range(3):
        cluster_id = int(hotspot_row[f"hotspot_{idx}"].iloc[0]) if not hotspot_row.empty and f"hotspot_{idx}" in hotspot_row else -1
        one_hot = np.zeros(max(n_clusters, 1), dtype=float)
        if 0 <= cluster_id < len(one_hot):
            one_hot[cluster_id] = 1.0
        values.extend(one_hot.tolist())

    stability_row = user_stability[user_stability["uid"].eq(uid)]
    for column in ["repeat_rate", "weekday_entropy", "holiday_entropy", "profile_diff"]:
        values.append(float(stability_row[column].iloc[0]) if not stability_row.empty and column in stability_row else 0.0)
    mobility_type = int(stability_row["mobility_type"].iloc[0]) if not stability_row.empty and "mobility_type" in stability_row else 0
    mobility_one_hot = np.zeros(4, dtype=float)
    if 0 <= mobility_type < 4:
        mobility_one_hot[mobility_type] = 1.0
    values.extend(mobility_one_hot.tolist())
    values.append(float((d % 7) in [5, 6]))
    weekday_one_hot = np.zeros(7, dtype=float)
    weekday_one_hot[d % 7] = 1.0
    values.extend(weekday_one_hot.tolist())
    return np.asarray(values, dtype=np.float32)


def build_condition_table(
    uids: list[int],
    days: list[int],
    user_hotspots: pd.DataFrame,
    user_stability: pd.DataFrame,
    grid_poi: pd.DataFrame,
    cluster_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build condition vectors for each (uid, d) pair."""
    if "cluster_id" in cluster_map and not cluster_map.empty:
        n_clusters = int(cluster_map.loc[cluster_map["cluster_id"].ge(0), "cluster_id"].nunique())
    else:
        n_clusters = 1
    rows = []
    for uid in uids:
        for d in days:
            rows.append(
                {
                    "uid": int(uid),
                    "d": int(d),
                    "condition": build_condition_vector(
                        uid=int(uid),
                        d=int(d),
                        user_hotspots=user_hotspots,
                        user_stability=user_stability,
                        grid_poi=grid_poi,
                        cluster_map=cluster_map,
                        n_clusters=max(n_clusters, 1),
                    ),
                }
            )
    return pd.DataFrame(rows)


class _TrajectoryDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, condition_df: pd.DataFrame):
        self.rows = []
        cond_lookup = {
            (int(row.uid), int(row.d)): np.asarray(row.condition, dtype=np.float32)
            for row in condition_df.itertuples(index=False)
        }
        for (uid, d), group in train_df.groupby(["uid", "d"], sort=False):
            if len(group) == 0:
                continue
            traj = group.sort_values("t")[["x", "y"]].to_numpy(dtype=np.float32)
            if len(traj) < TIME_STEPS:
                padded = np.repeat(traj[-1:], TIME_STEPS, axis=0)
                padded[: len(traj)] = traj
                traj = padded
            traj = traj[:TIME_STEPS] / (GRID_SIZE - 1)
            condition = cond_lookup.get((int(uid), int(d)))
            if condition is not None:
                self.rows.append((traj, condition))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        traj, condition = self.rows[index]
        return torch.tensor(traj, dtype=torch.float32), torch.tensor(condition, dtype=torch.float32)


def train_cvae(
    train_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    model: CVAE,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1.0,
    checkpoint_path: str = "models/cvae_checkpoint.pt",
) -> CVAE:
    """Train a CVAE model from daily trajectories and condition vectors."""
    dataset = _TrajectoryDataset(train_df, condition_df)
    if len(dataset) == 0:
        raise ValueError("No CVAE training samples were built")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, condition in loader:
            x = x.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x, condition)
            loss = cvae_loss(x_recon, x, mu, log_var, beta)
            loss.backward()
            optimizer.step()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def predict_trajectories(model: CVAE, condition_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    """Generate trajectories for all condition rows matching test_days."""
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for row in condition_df.itertuples(index=False):
            if int(row.d) not in test_days:
                continue
            condition = torch.tensor(np.asarray(row.condition, dtype=np.float32), device=device).unsqueeze(0)
            sample = model.sample(condition).squeeze(0).cpu().numpy()
            coords = np.rint(sample * (GRID_SIZE - 1)).clip(0, GRID_SIZE - 1).astype(int)
            for t, (x, y) in enumerate(coords):
                rows.append({"uid": int(row.uid), "d": int(row.d), "t": t, "x": int(x), "y": int(y)})
    return pd.DataFrame(rows, columns=["uid", "d", "t", "x", "y"])
