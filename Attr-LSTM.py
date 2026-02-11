# -*- coding: utf-8 -*-
"""
CAMELS - Pure LSTM baseline (multi-basin, daily) — 8-step direct multi-horizon
configuration aligned with Wang et al. (2025) MCR-LSTM baseline style.

Key choices:
- Sequence length: 365 days
- Target: discharge at time t...t+7 (lead_day = 0~7)
  - lead_day = 0 : same day as last input in the window (t)
  - lead_day = 1~7 : next 7 days
- Inputs: 5 NLDAS forcings + 27 static attributes, all globally standardized
- Output: 8-dim discharge vector q_mm, globally standardized (no log1p)
- Loss: NSE-style loss with 1 / (q_std_basin + eps)^2 weighting,
        averaged over all 8 lead days
- LSTM: 1 layer, hidden_size=256, dropout=0, initial forget-gate bias=3
- LR scheduler: ReduceLROnPlateau (monitoring val mean NSE over all leads)
- Early stopping: based on val mean NSE (all leads), with patience
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ================== CONFIG ==================


class Config:
    ROOT = Path(r"/home/oydh/USA_LSTM")

    OUT_DIR = ROOT / "camels_lstm_runs"
    RUN_NAME = "case0_2"

    STREAMFLOW_DIR = ROOT / "basin_dataset_public_v1p2" / "usgs_streamflow"
    FORCING_DIR = ROOT / "basin_dataset_public_v1p2" / "basin_mean_forcing" / "nldas"
    STATIC_DIR = ROOT / "static_attri"

    BASIN_CSV = ROOT / "basin_list.csv"
    USE_BASIN_CSV = True


    TRAIN_START = "1999-10-01"
    TRAIN_END = "2008-09-30"
    VAL_START = "1989-10-01"
    VAL_END = "1999-09-30"

    # NLDAS
    FORCING_VARS = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

    # Here we only select the attributes that correspond to the raster data.
    STATIC_VARS = [
        "p_mean",  "p_seasonality",
        "pet_mean", "aridity",
        #"frac_snow",
        "frac_forest",
        #"high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
        "elev_mean", "slope_mean",
        "area_gages2",
        #"lai_max", "lai_diff","gvf_max", "gvf_diff",
        "soil_depth_pelletier", "soil_depth_statsgo","soil_porosity", "soil_conductivity",
        #"carbonate_rocks_frac","geol_permeability",
        "max_water_content",
        "sand_frac", "silt_frac", "clay_frac",
    ]


    SEQ_LEN = 365
    HORIZON = 1              # lead_day = 0~7  1-8
    BATCH_SIZE = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    DROPOUT = 0
    INITIAL_FORGET_BIAS = 3.0
    LR_BASE = 1e-3
    MAX_EPOCHS = 50
    PATIENCE = 8             # early stopping patience (by val mean NSE over all leads)
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ReduceLROnPlateau
    LR_FACTOR = 0.5
    LR_PATIENCE = 2
    MIN_LR = 1e-5
    # === Random seed for reproducibility ===
    SEED = 2025

CFG = Config()


def set_seeds(seed: int = 42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU / GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================== Global normalization (directly adopting NLDAS SCALER) ==================

SCALER = {
    # NLDAS Input global mean / standard deviation
    "input_means": np.array([3.015, 357.68, 10.864, 10.864, 1055.533],
                            dtype=np.float32),
    "input_stds": np.array([7.573, 129.878, 10.932, 10.932, 705.998],
                           dtype=np.float32),
    # The mean value / standard deviation of the original traffic volume (in mm/day)
    "output_mean": np.array([1.49996196], dtype=np.float32),
    "output_std": np.array([3.62443672], dtype=np.float32),
}


def normalize_inputs(x: np.ndarray) -> np.ndarray:

    return (x - SCALER["input_means"]) / SCALER["input_stds"]


def normalize_target_q(q: np.ndarray) -> np.ndarray:

    return (q - SCALER["output_mean"][0]) / SCALER["output_std"][0]


def denormalize_target_q(q_norm: np.ndarray) -> np.ndarray:

    return q_norm * SCALER["output_std"][0] + SCALER["output_mean"][0]


# ================== evaluation index ==================


def nse(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((sim - obs) ** 2) / denom


def kge(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else np.nan
    cv_s = sim.std() / sim.mean() if sim.mean() != 0 else np.nan
    cv_o = obs.std() / obs.mean() if obs.mean() != 0 else np.nan
    gamma = cv_s / cv_o if cv_o != 0 else np.nan
    return 1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def pearson_r(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) == 0:
        return np.nan
    return np.corrcoef(obs, sim)[0, 1]


def fhv_percent_bias(sim, obs, frac=0.02):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) == 0:
        return np.nan
    order = np.argsort(-obs)
    sim_sorted, obs_sorted = sim[order], obs[order]
    n = max(1, int(len(obs_sorted) * frac))
    sim_h, obs_h = sim_sorted[:n], obs_sorted[:n]
    denom = np.sum(obs_h)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(sim_h - obs_h) / denom


def flv_percent_bias(sim, obs, frac=0.3):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) == 0:
        return np.nan
    order = np.argsort(obs)
    sim_sorted, obs_sorted = sim[order], obs[order]
    n = max(1, int(len(obs_sorted) * frac))
    sim_l, obs_l = sim_sorted[:n], obs_sorted[:n]
    denom = np.sum(obs_l)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(sim_l - obs_l) / denom


def pbias(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) == 0:
        return np.nan
    denom = np.sum(obs)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / denom


# ================== data reading ==================


def find_camels_file(root: Path, gauge_id: str, suffix: str) -> Path:
    matches = list(root.rglob(f"{gauge_id}{suffix}"))
    if len(matches) == 0:
        raise FileNotFoundError(f"Can't find out files: {root} *{gauge_id}{suffix}")
    if len(matches) > 1:
        print(f"WARNING: Multiple matches for {gauge_id}{suffix} were found. Defaulting to the first one: {matches[0]}")
    return matches[0]


def read_static_attributes(static_dir: Path, static_vars: List[str]) -> pd.DataFrame:
    files = list(static_dir.glob("camels_*.txt"))
    if not files:
        raise FileNotFoundError(f"No camels_*.txt in {static_dir}")

    df_all = None
    for fp in files:
        tmp = pd.read_csv(fp, sep=";")
        tmp["gauge_id"] = tmp["gauge_id"].astype(str).str.zfill(8)
        tmp = tmp.set_index("gauge_id")
        if df_all is None:
            df_all = tmp
        else:
            df_all = df_all.join(tmp, how="outer", rsuffix="_dup")

    needed = list(static_vars)
    if "area_gages2" not in needed:
        needed.append("area_gages2")

    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        print("WARNING: missing static attributes:", missing)

    cols = [c for c in needed if c in df_all.columns]
    df_all = df_all[cols].astype(float)
    return df_all


def read_streamflow(basin_id: str, area_km2: float, streamflow_dir: Path) -> pd.Series:
    fp = find_camels_file(streamflow_dir, basin_id, "_streamflow_qc.txt")
    cols = ["gauge_id", "year", "month", "day", "q_cfs", "flag"]
    df = pd.read_csv(fp, sep=r"\s+", header=None, names=cols)

    dates = pd.to_datetime(dict(
        year=df["year"],
        month=df["month"],
        day=df["day"],
    ))
    q_cfs = df["q_cfs"].astype(float)
    q_cfs = q_cfs.where(q_cfs >= 0.0, np.nan)

    factor = 0.0283168466 * 86400 * 1000 / (area_km2 * 1e6)  # cfs -> mm/day
    q_mm = q_cfs * factor
    return pd.Series(q_mm.values, index=dates, name="q_mm")


def read_forcing(basin_id: str, forcing_dir: Path, vars_use: List[str]) -> pd.DataFrame:
    fp = find_camels_file(forcing_dir, basin_id, "_lump_nldas_forcing_leap.txt")
    df = pd.read_csv(fp, sep=r"\s+", skiprows=3)
    if "Year" not in df.columns:
        raise RuntimeError(f"'Year' column not in forcing file {fp}")
    dates = pd.to_datetime(dict(year=df["Year"], month=df["Mnth"], day=df["Day"]))
    df.index = dates

    missing = [v for v in vars_use if v not in df.columns]
    if missing:
        raise RuntimeError(f"Forcing file {fp} missing cols {missing}")

    return df[vars_use].astype(float)


def align_forcing_flow(forc: pd.DataFrame, q: pd.Series) -> pd.DataFrame:
    df = forc.join(q, how="inner")
    return df.dropna()


# ================== NSE-style Loss（多步版） ==================


class MultiStepNSELoss(nn.Module):


    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor) -> torch.Tensor:
        # y_pred, y_true: [B, H]
        # q_stds: [B]
        se = (y_pred - y_true) ** 2  # [B,H]
        weights = 1.0 / (q_stds + self.eps) ** 2  # [B]
        weights = weights.unsqueeze(-1)  # [B,1]
        loss = torch.mean(weights * se)  # 对 B 和 H 都取平均
        return loss


# ================== 模型 ==================


class LSTMForecast(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        out_horizon: int,
        initial_forget_bias: float = 3.0,
    ):
        super().__init__()
        self.horizon = out_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, out_horizon)

        # 初始化 forget gate bias
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0) // 4
                bias.data[n:2 * n].fill_(initial_forget_bias)

    def forward(self, x):
        out, _ = self.lstm(x)          # [B, T, H]
        h_last = out[:, -1, :]         # [B, H]
        y = self.fc(h_last)            # [B, horizon]
        return y                       # [B, horizon]


# ================== Dataset (多步版) ==================


class CamelsMultiStepDataset(Dataset):


    def __init__(
        self,
        series_by_basin: Dict[str, pd.DataFrame],
        static_scaled: pd.DataFrame,
        seq_len: int,
        horizon: int,
        q_std_by_basin: Dict[str, float],
        target_start: pd.Timestamp,
        target_end: pd.Timestamp,
    ):
        self.series_by_basin = series_by_basin
        self.static_scaled = static_scaled
        self.seq_len = seq_len
        self.horizon = horizon
        self.q_std_by_basin = q_std_by_basin
        self.target_start = pd.to_datetime(target_start)
        self.target_end = pd.to_datetime(target_end)

        self.samples = []  # list of (basin_id, end_idx)

        self._build_index()

    def _build_index(self):
        H = self.horizon
        for bid, df in self.series_by_basin.items():
            n = len(df)
            if n < self.seq_len + H - 1:
                continue
            times = df.index

            for end_idx in range(self.seq_len - 1, n - H + 1):
                t = times[end_idx]
                t_last = times[end_idx + H - 1]
                if t < self.target_start or t > self.target_end:
                    continue
                if t_last > self.target_end:
                    continue
                self.samples.append((bid, end_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basin_id, end_idx = self.samples[idx]
        df = self.series_by_basin[basin_id]

        start_idx = end_idx - self.seq_len + 1
        hist = df.iloc[start_idx:end_idx + 1]

        fut = df.iloc[end_idx:end_idx + self.horizon]
        q_seq = fut["q_mm"].values.astype(np.float32)  # [H]

        x_dyn = hist[CFG.FORCING_VARS].values.astype(np.float32)
        x_dyn = normalize_inputs(x_dyn)  # [T,5]

        x_static = self.static_scaled.loc[basin_id, CFG.STATIC_VARS].values.astype(np.float32)
        T = x_dyn.shape[0]
        x_static_rep = np.repeat(x_static[None, :], T, axis=0)  # [T,27]

        x_all = np.concatenate([x_dyn, x_static_rep], axis=1).astype(np.float32)  # [T, 32]

        y = normalize_target_q(q_seq)  # [H]

        q_std = self.q_std_by_basin[basin_id]  # 原始 mm/day 标准差

        x_all = torch.from_numpy(x_all)
        y = torch.from_numpy(y.astype(np.float32))       # [H]
        q_std = torch.tensor(q_std, dtype=torch.float32)

        return x_all, y, basin_id, q_std


# ================== data preparation ==================


def prepare_series_and_scalers(
    basin_ids: List[str],
    static_all: pd.DataFrame,
    cfg: Config,
):
    train_start = pd.to_datetime(cfg.TRAIN_START)
    train_end = pd.to_datetime(cfg.TRAIN_END)
    val_start = pd.to_datetime(cfg.VAL_START)
    val_end = pd.to_datetime(cfg.VAL_END)

    static_sub = static_all.loc[basin_ids].copy()
    if "area_gages2" not in static_sub.columns:
        raise RuntimeError("STATIC_VARS must include area_gages2 for unit conversion")

    basin_reason = {bid: "not_processed" for bid in basin_ids}
    series_train: Dict[str, pd.DataFrame] = {}
    series_val: Dict[str, pd.DataFrame] = {}
    q_std_by_basin: Dict[str, float] = {}

    static_train_ids: List[str] = []

    print("Reading forcings & streamflow for all basins ...")
    for bid in tqdm(basin_ids):
        area_km2 = float(static_sub.loc[bid, "area_gages2"])
        try:
            q_series = read_streamflow(bid, area_km2, cfg.STREAMFLOW_DIR)
            forc = read_forcing(bid, cfg.FORCING_DIR, cfg.FORCING_VARS)
        except FileNotFoundError:
            basin_reason[bid] = "missing_stream_or_forcing"
            continue

        df = align_forcing_flow(forc, q_series)
        if df.empty:
            basin_reason[bid] = "no_overlap_or_all_nan"
            continue

        mask = (df.index >= (val_start - pd.Timedelta(days=cfg.SEQ_LEN - 1))) & (df.index <= train_end)
        df = df.loc[mask].copy()
        if df.empty:
            basin_reason[bid] = "no_data_in_train_val_window"
            continue

        df_train_target = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
        df_train_full = df.loc[(df.index >= (train_start - pd.Timedelta(days=cfg.SEQ_LEN - 1))) &
                               (df.index <= train_end)].copy()

        df_val_full = df.loc[(df.index >= (val_start - pd.Timedelta(days=cfg.SEQ_LEN - 1))) &
                             (df.index <= val_end)].copy()
        df_val_target = df.loc[(df.index >= val_start) & (df.index <= val_end)].copy()

        has_train = len(df_train_target) >= cfg.SEQ_LEN + cfg.HORIZON - 1
        has_val = len(df_val_target) >= cfg.SEQ_LEN + cfg.HORIZON - 1

        if has_train:
            series_train[bid] = df_train_full
            q_std_by_basin[bid] = float(df_train_target["q_mm"].std(ddof=0))
            static_train_ids.append(bid)

            if has_val:
                series_val[bid] = df_val_full
            else:
                basin_reason[bid] = "ok_for_train_but_short_val"
        else:
            if has_val:
                basin_reason[bid] = "no_train_but_has_val"
            else:
                basin_reason[bid] = "too_short_for_train_and_val"

    if len(series_train) == 0:
        raise RuntimeError("No basin has enough training data with given SEQ_LEN and HORIZON.")

    static_train = static_sub.loc[static_train_ids, cfg.STATIC_VARS].astype(float)
    mean_s = static_train.mean(axis=0)
    std_s = static_train.std(axis=0).replace(0, 1.0)

    static_scaled = static_sub.copy()
    static_scaled[cfg.STATIC_VARS] = (static_scaled[cfg.STATIC_VARS] - mean_s) / std_s

    return series_train, series_val, static_scaled, basin_reason, q_std_by_basin


# ================== Multi-step verification (lead 0 to 7) ==================


def eval_multi_step(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    horizon: int,
) -> Tuple[float, List[float], pd.DataFrame]:

    model.eval()

    basin_preds = {h: defaultdict(list) for h in range(horizon)}
    basin_obs = {h: defaultdict(list) for h in range(horizon)}

    with torch.no_grad():
        for X, y_norm, basin_ids, q_stds in tqdm(loader, desc="Eval (multi-step)", leave=False):
            X = X.to(device)                     # [B,T,D]
            y_norm = y_norm.to(device)           # [B,H]
            y_hat_norm = model(X)                # [B,H]

            y_hat_q = denormalize_target_q(y_hat_norm.cpu().numpy())  # [B,H]
            y_q = denormalize_target_q(y_norm.cpu().numpy())          # [B,H]

            B = y_q.shape[0]
            for i in range(B):
                bid = str(basin_ids[i])
                for h in range(horizon):
                    basin_preds[h][bid].append(y_hat_q[i, h])
                    basin_obs[h][bid].append(y_q[i, h])

    rows = []
    mean_nse_by_lead = []

    for h in range(horizon):
        all_nse = []
        for bid in basin_preds[h].keys():
            sim = np.array(basin_preds[h][bid])
            obs = np.array(basin_obs[h][bid])

            met = {
                "basin_id": bid,
                "lead_day": h,
                "NSE": nse(sim, obs),
                "KGE": kge(sim, obs),
                "Pearson_r": pearson_r(sim, obs),
                "FHV_0.02": fhv_percent_bias(sim, obs, frac=0.02),
                "FHV_0.05": fhv_percent_bias(sim, obs, frac=0.05),
                "FHV_0.10": fhv_percent_bias(sim, obs, frac=0.10),
                "FLV_0.30": flv_percent_bias(sim, obs, frac=0.30),
                "PBIAS": pbias(sim, obs),
            }
            rows.append(met)
            all_nse.append(met["NSE"])

        mean_nse = np.nanmean(all_nse) if len(all_nse) > 0 else np.nan
        mean_nse_by_lead.append(mean_nse)

    df_metrics = pd.DataFrame(rows)
    mean_NSE_global = np.nanmean(mean_nse_by_lead) if len(mean_nse_by_lead) > 0 else np.nan
    return mean_NSE_global, mean_nse_by_lead, df_metrics


# ================== 训练主流程 ==================


def train_and_eval():
    cfg = CFG
    # === 固定随机种子 ===
    set_seeds(cfg.SEED)

    device = cfg.DEVICE
    print("Using device:", device)

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.OUT_DIR / cfg.RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Outputs will be saved to:", run_dir)

    # 1) static attribute
    static_all = read_static_attributes(cfg.STATIC_DIR, cfg.STATIC_VARS)

    # 2) List of river basins
    if cfg.USE_BASIN_CSV and cfg.BASIN_CSV is not None and cfg.BASIN_CSV.exists():
        df_ids = pd.read_csv(cfg.BASIN_CSV)
        basin_ids = df_ids["gauge_id"].astype(str).str.zfill(8).tolist()
    else:
        basin_ids = static_all.index.tolist()

    # 3) Preparation sequence & Static scaling & q_std
    series_train, series_val, static_scaled, basin_reason, q_std_by_basin = prepare_series_and_scalers(
        basin_ids, static_all, cfg
    )

    # Output areas that cannot be included in the training process
    trainable_ids = set(series_train.keys())
    cannot_train = []
    for bid in basin_ids:
        if bid not in trainable_ids:
            reason = basin_reason.get(bid, "unknown")
            cannot_train.append({"gauge_id": bid, "reason": reason})
    if cannot_train:
        miss_df = pd.DataFrame(cannot_train).sort_values("gauge_id")
        miss_path = run_dir / "basins_cannot_train.csv"
        miss_df.to_csv(miss_path, index=False)
        print(f"{len(cannot_train)} basins cannot be used for training. Details saved to:", miss_path)
    else:
        print("All candidate basins have enough data for training.")

    # 4) Dataset & DataLoader
    train_ds = CamelsMultiStepDataset(
        series_by_basin=series_train,
        static_scaled=static_scaled,
        seq_len=cfg.SEQ_LEN,
        horizon=cfg.HORIZON,
        q_std_by_basin=q_std_by_basin,
        target_start=pd.to_datetime(cfg.TRAIN_START),
        target_end=pd.to_datetime(cfg.TRAIN_END),
    )
    val_ds = CamelsMultiStepDataset(
        series_by_basin=series_val,
        static_scaled=static_scaled,
        seq_len=cfg.SEQ_LEN,
        horizon=cfg.HORIZON,
        q_std_by_basin=q_std_by_basin,
        target_start=pd.to_datetime(cfg.VAL_START),
        target_end=pd.to_datetime(cfg.VAL_END),
    )

    # === Set a fixed random generator for the DataLoader (to control the shuffle sequence) ===
    g = torch.Generator()
    g.manual_seed(cfg.SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        drop_last=False,
    )

    in_dim = len(cfg.FORCING_VARS) + len(cfg.STATIC_VARS)
    model = LSTMForecast(
        input_size=in_dim,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        out_horizon=cfg.HORIZON,
        initial_forget_bias=cfg.INITIAL_FORGET_BIAS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR_BASE)
    loss_fn = MultiStepNSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE,
        min_lr=cfg.MIN_LR,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_mean_NSE_global": [],
        "val_mean_NSE_lead0": [],
    }
    best_val_nse = -np.inf
    best_state = None
    patience = cfg.PATIENCE

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        # ------- Train -------
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for X, y_norm, basin_ids, q_stds in pbar:
            X = X.to(device)              # [B,T,D]
            y_norm = y_norm.to(device)    # [B,H]
            q_stds = q_stds.to(device)    # [B]

            optimizer.zero_grad()
            y_hat_norm = model(X)         # [B,H]

            loss = loss_fn(y_hat_norm, y_norm, q_stds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(train_losses))

        train_loss_mean = float(np.mean(train_losses))

        # ------- Val/Test -------
        val_mean_nse_global, mean_nse_by_lead, _ = eval_multi_step(
            model=model,
            loader=val_loader,
            device=device,
            horizon=cfg.HORIZON,
        )

        val_mean_nse_lead0 = mean_nse_by_lead[0] if len(mean_nse_by_lead) > 0 else np.nan

        print(
            f"Epoch {epoch}: train_loss={train_loss_mean:.4f}, "
            f"val_mean_NSE_global={val_mean_nse_global:.3f}, "
            f"val_mean_NSE_lead0={val_mean_nse_lead0:.3f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_mean)
        history["val_mean_NSE_global"].append(val_mean_nse_global)
        history["val_mean_NSE_lead0"].append(val_mean_nse_lead0)

        # Adjust the learning rate (automatically decrease)
        scheduler.step(val_mean_nse_global)

        # Record the best model + Early stopping
        if val_mean_nse_global > best_val_nse + 1e-4:
            best_val_nse = val_mean_nse_global
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_nse_global": best_val_nse,
                "config": cfg.__dict__,
            }
            torch.save(best_state, run_dir / "camels_lstm_multistep8_best.pt")
            patience = cfg.PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # If best_state has never been updated, then save the result of the last round.
    if best_state is None:
        best_state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_nse_global": best_val_nse,
            "config": cfg.__dict__,
        }
        torch.save(best_state, run_dir / "camels_lstm_multistep8_last.pt")
        print("No improvement observed; saved last epoch model.")
    else:
        print("Best model saved to:", run_dir / "camels_lstm_multistep8_best.pt")

    # Save the training curve
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / "train_history_multistep8.csv", index=False)

    # Draw the loss & validation NSE curve
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        hist_df["epoch"], hist_df["val_mean_NSE_global"],
        marker="s", linestyle="-", label="Val mean NSE (all leads)", color="tab:orange"
    )
    ax2.set_ylabel("Val mean NSE (all leads)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")

    plt.title("Training Loss & Validation mean NSE (0–7 days)")
    plt.tight_layout()
    plt.savefig(run_dir / "train_val_curve_multistep8.png", dpi=200)
    plt.close(fig)

    # ===== Re-calculate the indicators of each lead using the best model and output the results as separate CSV files for each lead =====
    print("Reloading best model and computing per-lead metrics ...")
    model.load_state_dict(best_state["model_state"])
    model.to(device)
    model.eval()

    final_mean_nse_global, final_mean_nse_by_lead, df_val_metrics_all = eval_multi_step(
        model=model,
        loader=val_loader,
        device=device,
        horizon=cfg.HORIZON,
    )

    print("Final mean_val_NSE_global (0–7 days) =", final_mean_nse_global)
    print("Mean NSE by lead_day (0..7):", final_mean_nse_by_lead)

    # Each "lead_day" is stored in a separate CSV file.
    for lead in range(cfg.HORIZON):
        df_lead = df_val_metrics_all[df_val_metrics_all["lead_day"] == lead].copy()
        out_path = run_dir / f"val_metrics_lead{lead}.csv"
        df_lead.to_csv(out_path, index=False)
        print(f"Saved metrics for lead_day={lead} to:", out_path)

    print("All done.")


if __name__ == "__main__":
    train_and_eval()
