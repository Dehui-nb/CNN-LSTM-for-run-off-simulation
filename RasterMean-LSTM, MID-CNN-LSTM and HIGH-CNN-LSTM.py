# -*- coding: utf-8 -*-
"""
CAMELS - LSTM + Static (TAB / LOW / MID / HIGH) with:
- MID-only Channel Attention (softmax over input TIF channels), optional switch
- HIGH patch pooling: mean vs attention pooling, optional switch
- LOW uses mean only (no std) to align with area-mean semantics
- Outputs additional metrics: FHV_0.02/0.05/0.10 and FLV_0.30

HIGH modification in this version:
- HIGH patches are COVERING (sliding-window tiling) over basin bbox,
  filtered by valid fraction, then deterministically sub-sampled to HIGH_MAX_PATCHES.
- NO random sampling for HIGH anymore (stable per basin).

Notes:
- Number of TIF channels C is inferred from TIF_LIST (not fixed)
- When USE_TAB=False: no CAMELS static attributes are input
- Dynamic forcings (5 vars) are always input
"""

import os
import re
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import rasterio.features

import matplotlib.pyplot as plt
import gc


# ================== CONFIG ==================

class Config:
    ROOT = Path(r"/home/oydh/USA_LSTM")
    OUT_DIR = ROOT / "camels_cnn_lstm_runs"
    RUN_NAME = "case3_new"

    STREAMFLOW_DIR = ROOT / "basin_dataset_public_v1p2" / "usgs_streamflow"
    FORCING_DIR = ROOT / "basin_dataset_public_v1p2" / "basin_mean_forcing" / "nldas"
    STATIC_DIR = ROOT / "static_attri"

    BASIN_CSV = ROOT / "basin_list.csv"
    USE_BASIN_CSV = True

    TRAIN_START = "1999-10-01"
    TRAIN_END   = "2008-09-30"
    VAL_START   = "1989-10-01"
    VAL_END     = "1999-09-30"

    FORCING_VARS = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

    #
    STATIC_VARS = [
        "area_gages2",
    ]

    # ====== sequence / horizon ======
    SEQ_LEN = 365
    HORIZON = 1

    # ====== training ======
    BATCH_SIZE = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    DROPOUT = 0.0
    INITIAL_FORGET_BIAS = 3.0
    LR_BASE = 1e-3
    MAX_EPOCHS = 30
    EARLY_STOP_PATIENCE = 8
    NUM_WORKERS = 0
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    SEED = 2025

    # ====== shapefile ======
    SHP_FILE = ROOT / "shp" / "HCDN_nhru_final_671.shp"
    BASIN_ID_FIELD = "hru_id"
    BASIN_TARGET_CRS = "EPSG:4326"

    # ====== aligned TIF list (C inferred from here) ======
    TIF_LIST = [
        ("dem", "/home/oydh/stream/USA_stream/static/usa_elv.tif"),
        ("slope", "/home/oydh/stream/USA_stream/static/usa_elv_slpe_nopro.tif"),
        ("tree", "/home/oydh/stream/USA_stream/static/resampled_to_dem/tree_to_dem.tif"),
        ("ai", "/home/oydh/stream/USA_stream/static/resampled_to_dem/ai_to_dem.tif"),
        ("et0", "/home/oydh/stream/USA_stream/static/resampled_to_dem/et0_to_dem.tif"),
        ("pre_mean", "/home/oydh/stream/USA_stream/static/resampled_to_dem/pre_mean_to_dem.tif"),
        ("pre_sea", "/home/oydh/stream/USA_stream/static/resampled_to_dem/pre_sea_to_dem.tif"),
        ("clay", "/home/oydh/stream/USA_stream/static/resampled_to_dem/clay_to_dem.tif"),
        ("sand", "/home/oydh/stream/USA_stream/static/resampled_to_dem/sand_to_dem.tif"),
        ("silt", "/home/oydh/stream/USA_stream/static/resampled_to_dem/silt_to_dem.tif"),
        ("soil_depth", "/home/oydh/stream/USA_stream/static/resampled_to_dem/soil_depth_to_dem.tif"),
        ("soil_depth2", "/home/oydh/stream/USA_stream/static/resampled_to_dem/soil_depth2_to_dem.tif"),
        ("porosity", "/home/oydh/stream/USA_stream/static/resampled_to_dem/porosity_to_dem.tif"),
        ("water", "/home/oydh/stream/USA_stream/static/resampled_to_dem/water_to_dem.tif"),
        ("conductivity", "/home/oydh/stream/USA_stream/static/resampled_to_dem/conductivity_to_dem.tif"),
    ]

    # ===== HIGH patches: COVERING (tiling) =====
    HIGH_PATCH_SIZE = 128         # P
    HIGH_STRIDE = 128             # S (S=P => non-overlap tiling)
    HIGH_MIN_VALID_FRAC = 0.6    # keep a patch if basin_mask fraction >= this
    HIGH_MAX_PATCHES = 20       # deterministic sub-sample cap
    HIGH_PAD_MODE = "reflect"    # padding when bbox < P

    # ====== four switches ======
    USE_TAB  = True
    USE_LOW  = True    # LOW = mean only (C dims)
    USE_MID  = True    # MID = fixed MID_SIZE x MID_SIZE -> encoder
    USE_HIGH = True   # HIGH = covering patches P x P -> encoder + pooling

    MID_SIZE = 64      # MID grid size (SxS)

    # ====== attention switches ======
    USE_MID_CHANNEL_ATT = True       # MID-only channel attention over input TIF channels (softmax)
    USE_HIGH_PATCH_ATT  = False      # HIGH patch attention pooling (else mean pooling)
    PATCH_ATT_HIDDEN    = 64         # hidden dim in patch attention MLP

    # ====== CNN embedding dim ======
    CNN_EMB_DIM = 32


CFG = Config()


# ================== seeds ==================

def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== Global scaler (your baseline) ==================

SCALER = {
    "input_means": np.array([3.015, 357.68, 10.864, 10.864, 1055.533], dtype=np.float32),
    "input_stds":  np.array([7.573, 129.878, 10.932, 10.932, 705.998], dtype=np.float32),
    "output_mean": np.array([1.49996196], dtype=np.float32),
    "output_std":  np.array([3.62443672], dtype=np.float32),
}

def normalize_inputs(x: np.ndarray) -> np.ndarray:
    return (x - SCALER["input_means"]) / SCALER["input_stds"]

def normalize_target_q(q: np.ndarray) -> np.ndarray:
    return (q - SCALER["output_mean"][0]) / SCALER["output_std"][0]

def denormalize_target_q(q_norm: np.ndarray) -> np.ndarray:
    return q_norm * SCALER["output_std"][0] + SCALER["output_mean"][0]


# ================== ID / SHP caching ==================

def normalize_id(value: str) -> Optional[str]:
    s = str(value)
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return None
    try:
        return str(int(digits))
    except Exception:
        return None

_BASIN_GEOMS: Optional[Dict[str, object]] = None

def get_basin_geoms(cfg: Config) -> Dict[str, object]:
    global _BASIN_GEOMS
    if _BASIN_GEOMS is not None:
        return _BASIN_GEOMS

    shp_file = cfg.SHP_FILE
    if not shp_file.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_file}")

    gdf_all = gpd.read_file(shp_file)
    if gdf_all.crs is None:
        gdf_all.set_crs("EPSG:4326", inplace=True)
    gdf_all = gdf_all.to_crs(cfg.BASIN_TARGET_CRS)

    gdf_all["_key"] = gdf_all[cfg.BASIN_ID_FIELD].map(normalize_id)
    geoms = {}
    for _, row in gdf_all.iterrows():
        k = row["_key"]
        if k is None:
            continue
        geoms[k] = row["geometry"]

    _BASIN_GEOMS = geoms
    return _BASIN_GEOMS


# ================== TIF footprint mean/std (across all basins footprint) ==================

TIF_GLOBAL_STATS: Dict[str, Tuple[float, float]] = {}

def compute_tif_global_stats(cfg: Config) -> Dict[str, Tuple[float, float]]:
    global TIF_GLOBAL_STATS
    if TIF_GLOBAL_STATS:
        return TIF_GLOBAL_STATS

    print("Computing basin-footprint global mean/std for static TIFs ...")
    geoms_map = get_basin_geoms(cfg)
    all_geoms = list(geoms_map.values())

    stats: Dict[str, Tuple[float, float]] = {}
    for name, tifp in cfg.TIF_LIST:
        tifp = Path(tifp)
        if not tifp.exists():
            raise FileNotFoundError(f"TIF not found for global stats: {tifp}")

        with rasterio.open(tifp) as src:
            gser = gpd.GeoSeries(all_geoms, crs=cfg.BASIN_TARGET_CRS).to_crs(src.crs)
            shapes = [mapping(g) for g in gser]

            out, _ = mask(src, shapes, crop=False, filled=False)
            masked_arr = out[0]  # MaskedArray
            data = masked_arr.data.astype(np.float32)
            msk = masked_arr.mask
            data[msk] = np.nan
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)

        m = float(np.nanmean(data))
        s = float(np.nanstd(data))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        stats[name] = (m, s)
        print(f"[FOOTPRINT STATS] {name}: mean={m:.4f}, std={s:.4f}")

    TIF_GLOBAL_STATS = stats
    return stats


# ================== Utilities ==================

def resize_with_mask_area(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Downsample/upsample arr(H,W) with NaN mask using area-average on valid pixels."""
    outH, outW = out_hw
    a = arr.astype(np.float32)
    m = np.isfinite(a).astype(np.float32)
    a = np.nan_to_num(a, nan=0.0)

    ta = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    tm = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)

    ta2 = F.interpolate(ta, size=(outH, outW), mode="area")
    tm2 = F.interpolate(tm, size=(outH, outW), mode="area")

    out = ta2 / torch.clamp(tm2, min=1e-6)
    out = out.squeeze(0).squeeze(0).numpy().astype(np.float32)
    out[tm2.squeeze(0).squeeze(0).numpy() < 1e-6] = 0.0
    return out


def _pad_to_min_hw(arr: np.ndarray, min_h: int, min_w: int, mode: str = "reflect", const_val: float = np.nan) -> np.ndarray:
    H, W = arr.shape
    padH = max(0, min_h - H)
    padW = max(0, min_w - W)
    if padH == 0 and padW == 0:
        return arr
    if mode == "reflect":
        # reflect requires size>1; if degenerate, fallback to constant
        if H < 2 or W < 2:
            return np.pad(arr, ((0, padH), (0, padW)), mode="constant", constant_values=const_val)
        return np.pad(arr, ((0, padH), (0, padW)), mode="reflect")
    return np.pad(arr, ((0, padH), (0, padW)), mode="constant", constant_values=const_val)


def _compute_bbox_window_and_basin_mask(
    src: rasterio.io.DatasetReader,
    geom_wgs84,
    target_crs: str,
):
    """Compute bbox window and basin mask (True inside basin) for that window."""
    geom_src = gpd.GeoSeries([geom_wgs84], crs=target_crs).to_crs(src.crs).iloc[0]
    win = rasterio.features.geometry_window(src, [mapping(geom_src)], pad_x=0, pad_y=0)
    transform = src.window_transform(win)

    # basin mask in window coords
    basin_mask = rasterio.features.geometry_mask(
        [mapping(geom_src)],
        out_shape=(win.height, win.width),
        transform=transform,
        invert=True,
        all_touched=False,
    )  # True inside basin

    return win, basin_mask


def _extract_patch(mat: np.ndarray, r: int, c: int, P: int, pad_mode: str) -> np.ndarray:
    patch = mat[r:r+P, c:c+P]
    if patch.shape != (P, P):
        patch = _pad_to_min_hw(patch, P, P, mode=pad_mode, const_val=np.nan)
    return patch


# ================== Static preprocessing: LOW/MID/HIGH ==================

def build_static_low_mid_high(
    basin_ids: List[str],
    cfg: Config,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], int, List[str]]:
    """
    Returns:
      static_low:  bid -> (C,)              mean per channel over valid pixels
      static_mid:  bid -> (C, MID, MID)     resized by area-average over valid pixels
      static_high: bid -> (K, C, P, P)      COVERING patches (tiling) with deterministic cap
      C: number of channels
      tif_names: list of TIF names in order
    """
    tif_stats = compute_tif_global_stats(cfg)
    geoms_map = get_basin_geoms(cfg)

    tif_paths = [(name, Path(p)) for name, p in cfg.TIF_LIST]
    for _, p in tif_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing TIF: {p}")

    tif_names = [n for n, _ in tif_paths]
    C = len(tif_paths)

    static_low: Dict[str, np.ndarray] = {}
    static_mid: Dict[str, np.ndarray] = {}
    static_high: Dict[str, np.ndarray] = {}

    P = int(cfg.HIGH_PATCH_SIZE)
    S = int(cfg.HIGH_STRIDE)
    min_frac = float(cfg.HIGH_MIN_VALID_FRAC)
    maxK = int(cfg.HIGH_MAX_PATCHES)

    print("Precomputing static LOW/MID/HIGH for basins ...")
    for bid in tqdm(basin_ids, desc="Static preprocess"):
        key = normalize_id(bid)
        if key is None or key not in geoms_map:
            continue
        geom = geoms_map[key]

        # ---- Step A: Use first TIF to define the bbox window + basin mask (stable across all TIFs) ----
        name0, tif0 = tif_paths[0]
        with rasterio.open(tif0) as src0:
            try:
                win, basin_mask = _compute_bbox_window_and_basin_mask(src0, geom, cfg.BASIN_TARGET_CRS)
            except Exception as e:
                warnings.warn(f"{bid}: geometry_window/mask failed on {name0}: {e}")
                continue

        H, W = basin_mask.shape
        if H == 0 or W == 0:
            continue

        # if bbox smaller than P, pad mask (and later pad arrays)
        basin_mask_pad = _pad_to_min_hw(basin_mask.astype(np.float32), P, P, mode="constant", const_val=0.0) > 0.5

        # ---- Step B: read all tif arrays using the SAME window, z-score, and apply basin mask as NaN outside ----
        crop_channels: List[np.ndarray] = []
        for name, tifp in tif_paths:
            with rasterio.open(tifp) as src:
                arr = src.read(1, window=win).astype(np.float32)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)

            # mask outside basin to NaN
            arr = np.where(basin_mask, arr, np.nan)

            m, s = tif_stats.get(name, (0.0, 1.0))
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            arr = (arr - m) / s

            # pad to at least P×P for safe patch extraction
            arr = _pad_to_min_hw(arr, P, P, mode=cfg.HIGH_PAD_MODE, const_val=np.nan)
            crop_channels.append(arr)

        # valid mask = inside basin AND finite for at least one channel? 这里用 basin_mask 即可
        # 但 LOW/MID 计算均值时仍以“该通道 finite”保证有效像元
        # basin_mask_pad is the padded basin mask
        valid_pix = int(basin_mask_pad.sum())

        # ---- LOW: mean per channel over valid pixels (finite inside basin) ----
        if cfg.USE_LOW:
            feats = []
            for ch in crop_channels:
                vals = ch[np.isfinite(ch)]
                mu = float(vals.mean()) if vals.size else 0.0
                feats.append(mu)
            static_low[bid] = np.array(feats, dtype=np.float32)  # (C,)

        # ---- MID: resize with area-average on valid pixels (NaN-aware) ----
        if cfg.USE_MID:
            mids = [resize_with_mask_area(ch, (cfg.MID_SIZE, cfg.MID_SIZE)) for ch in crop_channels]
            static_mid[bid] = np.stack(mids, axis=0).astype(np.float32)  # (C, MID, MID)

        # ---- HIGH: COVERING tiling patches over bbox ----
        if cfg.USE_HIGH:
            HH, WW = basin_mask_pad.shape

            rows = list(range(0, max(1, HH - P + 1), S))
            cols = list(range(0, max(1, WW - P + 1), S))

            valid_windows: List[Tuple[int, int]] = []
            for r in rows:
                for c in cols:
                    mwin = _extract_patch(basin_mask_pad.astype(np.float32), r, c, P, pad_mode="constant") > 0.5
                    frac = float(mwin.mean()) if mwin.size > 0 else 0.0
                    if frac >= min_frac:
                        valid_windows.append((r, c))

            if len(valid_windows) == 0:
                # fallback: center window
                cr = max(0, (HH - P) // 2)
                cc = max(0, (WW - P) // 2)
                valid_windows = [(cr, cc)]

            # deterministic cap
            if len(valid_windows) > maxK:
                idx = np.linspace(0, len(valid_windows) - 1, maxK).astype(int)
                valid_windows = [valid_windows[i] for i in idx]

            K = len(valid_windows)
            patches = np.zeros((K, C, P, P), dtype=np.float32)
            for ki, (r, c) in enumerate(valid_windows):
                for ci, ch in enumerate(crop_channels):
                    pch = _extract_patch(ch, r, c, P, pad_mode=cfg.HIGH_PAD_MODE)
                    pch = np.nan_to_num(pch, nan=0.0).astype(np.float32)
                    patches[ki, ci] = pch

            static_high[bid] = patches  # (K,C,P,P)

    return static_low, static_mid, static_high, C, tif_names


# ================== CAMELS I/O ==================

def find_camels_file(root: Path, gauge_id: str, suffix: str) -> Path:
    matches = list(root.rglob(f"{gauge_id}{suffix}"))
    if len(matches) == 0:
        raise FileNotFoundError(f"File not found: {root}/*{gauge_id}{suffix}")
    if len(matches) > 1:
        print(f"WARNING: multiple matches; use first: {matches[0]}")
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
        df_all = tmp if df_all is None else df_all.join(tmp, how="outer", rsuffix="_dup")

    needed = list(static_vars)
    if "area_gages2" not in needed:
        needed.append("area_gages2")

    cols = [c for c in needed if c in df_all.columns]
    miss = [c for c in needed if c not in df_all.columns]
    if miss:
        print("WARNING: missing static attributes:", miss)

    return df_all[cols].astype(float)

def read_streamflow(basin_id: str, area_km2: float, streamflow_dir: Path) -> pd.Series:
    fp = find_camels_file(streamflow_dir, basin_id, "_streamflow_qc.txt")
    cols = ["gauge_id", "year", "month", "day", "q_cfs", "flag"]
    df = pd.read_csv(fp, sep=r"\s+", header=None, names=cols)

    dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    q_cfs = df["q_cfs"].astype(float).where(lambda s: s >= 0.0, np.nan)

    factor = 0.0283168466 * 86400 * 1000 / (area_km2 * 1e6)
    q_mm = q_cfs * factor
    return pd.Series(q_mm.values, index=dates, name="q_mm")

def read_forcing(basin_id: str, forcing_dir: Path, vars_use: List[str]) -> pd.DataFrame:
    fp = find_camels_file(forcing_dir, basin_id, "_lump_nldas_forcing_leap.txt")
    df = pd.read_csv(fp, sep=r"\s+", skiprows=3)
    dates = pd.to_datetime(dict(year=df["Year"], month=df["Mnth"], day=df["Day"]))
    df.index = dates

    missing = [v for v in vars_use if v not in df.columns]
    if missing:
        raise RuntimeError(f"Forcing file {fp} missing cols {missing}")

    return df[vars_use].astype(float)

def align_forcing_flow(forc: pd.DataFrame, q: pd.Series) -> pd.DataFrame:
    return forc.join(q, how="inner").dropna()


# ================== Metrics ==================

def nse(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((sim - obs) ** 2) / denom

def kge(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
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
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(obs) == 0:
        return np.nan
    return np.corrcoef(obs, sim)[0, 1]

def pbias(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(obs) == 0:
        return np.nan
    denom = np.sum(obs)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / denom

def fhv_percent_bias(sim, obs, frac=0.02):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
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
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
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


# ================== MID-only Channel Attention (softmax over input channels) ==================

class MidInputChannelAttention(nn.Module):
    """
    Produces per-sample channel weights w (B,C), sum=1, using global avg pooling on MID input.
    Then scales input channels: x' = x * w[:, :, None, None]
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,C,H,W)
        s = x.mean(dim=(2, 3))                 # (B,C)
        w = torch.softmax(s, dim=1)            # (B,C), sum=1
        xw = x * w.unsqueeze(-1).unsqueeze(-1) # (B,C,H,W)
        return xw, w


# ================== CNN encoder ==================

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pyramid_levels=(1, 2, 4)):
        super().__init__()
        self.levels = tuple(int(l) for l in pyramid_levels)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d((l, l)) for l in self.levels])

    def forward(self, x):
        outs = [p(x).view(x.size(0), -1) for p in self.pools]
        return torch.cat(outs, dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))

class StaticEncoderSPP(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int, pyramid_levels=(1, 2, 4)):
        super().__init__()
        self.emb_dim = emb_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.backbone = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=2),
        )
        self.spp = SpatialPyramidPooling(pyramid_levels)
        feat_mul = sum(l * l for l in pyramid_levels)
        self.head = nn.Sequential(
            nn.Linear(256 * feat_mul, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.backbone(h)
        h = self.spp(h)
        return self.head(h)


# ================== HIGH Patch Attention Pooling ==================

class PatchAttentionPooling(nn.Module):
    """
    Attention pooling over K patch embeddings.
    Input: z (B,K,D)
    Output: pooled (B,D), weights (B,K)
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: (B,K,D)
        a = self.net(z).squeeze(-1)     # (B,K)
        w = torch.softmax(a, dim=1)     # (B,K)
        pooled = (z * w.unsqueeze(-1)).sum(dim=1)  # (B,D)
        return pooled, w


# ================== Loss ==================

class MultiStepNSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor) -> torch.Tensor:
        se = (y_pred - y_true) ** 2
        weights = 1.0 / (q_stds + self.eps) ** 2
        weights = weights.unsqueeze(-1)
        return torch.mean(weights * se)


# ================== Model ==================

class LSTMForecastStatic(nn.Module):
    def __init__(
        self,
        dyn_input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        static_in_ch: int,
        cnn_emb_dim: int,
        static_total_dim: int,
        horizon: int,
        cfg: Config,
        initial_forget_bias: float = 3.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.static_total_dim = static_total_dim
        self.horizon = horizon
        self.cfg = cfg
        self.static_in_ch = static_in_ch
        self.cnn_emb_dim = cnn_emb_dim

        self.mid_chan_att = MidInputChannelAttention(static_in_ch) if cfg.USE_MID_CHANNEL_ATT else None
        self.static_enc = StaticEncoderSPP(in_channels=static_in_ch, emb_dim=cnn_emb_dim)

        self.patch_pool = PatchAttentionPooling(cnn_emb_dim, cfg.PATCH_ATT_HIDDEN) if cfg.USE_HIGH_PATCH_ATT else None

        self.lstm = nn.LSTM(
            input_size=dyn_input_size + static_total_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0) // 4
                bias.data[n:2 * n].fill_(initial_forget_bias)

    def encode_mid(self, x_mid: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x_mid: (B,C,M,M)
        returns: z_mid (B,emb), w_channel (B,C) or None
        """
        if self.mid_chan_att is not None:
            x_mid, w = self.mid_chan_att(x_mid)
        else:
            w = None
        z = self.static_enc(x_mid)
        return z, w

    def encode_high(self, x_high: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x_high: (B,K,C,P,P)
        returns: z_high (B,emb), w_patch (B,K) or None
        """
        B, K, C, P, _ = x_high.shape
        x = x_high.view(B * K, C, P, P)
        z_each = self.static_enc(x).view(B, K, -1)  # (B,K,emb)

        if self.patch_pool is not None:
            z_pool, w_patch = self.patch_pool(z_each)
        else:
            z_pool = z_each.mean(dim=1)
            w_patch = None
        return z_pool, w_patch

    def forward_concat(self, x_dyn: torch.Tensor, static_vec: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_dyn.shape
        z_rep = static_vec.unsqueeze(1).expand(B, T, self.static_total_dim)
        x_all = torch.cat([x_dyn, z_rep], dim=-1)

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x_dyn.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x_dyn.device)

        out, _ = self.lstm(x_all, (h0, c0))
        h_last = out[:, -1, :]
        return self.fc(h_last)


# ================== Dataset ==================

class CamelsMultiStepDataset(Dataset):
    def __init__(
        self,
        series_by_basin: Dict[str, pd.DataFrame],
        seq_len: int,
        horizon: int,
        q_std_by_basin: Dict[str, float],
        target_start: pd.Timestamp,
        target_end: pd.Timestamp,
        forcing_vars: List[str],
    ):
        self.series_by_basin = series_by_basin
        self.seq_len = seq_len
        self.horizon = horizon
        self.q_std_by_basin = q_std_by_basin
        self.target_start = pd.to_datetime(target_start)
        self.target_end = pd.to_datetime(target_end)
        self.forcing_vars = forcing_vars

        self.samples: List[Tuple[str, int]] = []
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
                if t < self.target_start or t > self.target_end:
                    continue
                self.samples.append((bid, end_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basin_id, end_idx = self.samples[idx]
        df = self.series_by_basin[basin_id]

        start_idx = end_idx - self.seq_len + 1
        hist = df.iloc[start_idx:end_idx + 1]
        fut  = df.iloc[end_idx:end_idx + self.horizon]

        x_dyn = hist[self.forcing_vars].values.astype(np.float32)
        x_dyn = normalize_inputs(x_dyn)

        q_seq = fut["q_mm"].values.astype(np.float32)
        y = normalize_target_q(q_seq)

        q_std = self.q_std_by_basin[basin_id]
        return (
            torch.from_numpy(x_dyn),
            torch.from_numpy(y.astype(np.float32)),
            str(basin_id),
            torch.tensor(q_std, dtype=torch.float32),
        )


# ================== TAB vectors ==================

def build_static_attr_vectors(static_all: pd.DataFrame, basin_ids: List[str], cfg: Config, run_dir: Path) -> Tuple[Dict[str, np.ndarray], int, List[str]]:
    tab_cols = [c for c in cfg.STATIC_VARS if c in static_all.columns]
    if len(tab_cols) == 0:
        raise RuntimeError("No STATIC_VARS found in static_all columns!")

    sub = static_all.loc[basin_ids, tab_cols].copy().astype(float)
    mean_vec = sub.mean(axis=0)
    std_vec  = sub.std(axis=0)
    std_vec[std_vec == 0.0] = 1.0
    sub_norm = (sub - mean_vec) / std_vec

    scaler_path = run_dir / "static_attr_scaler.npz"
    np.savez(
        scaler_path,
        cols=np.array(tab_cols, dtype="U"),
        mean=mean_vec.values.astype(np.float32),
        std=std_vec.values.astype(np.float32),
    )
    print(f"Static attribute scaler saved to: {scaler_path}")

    tab_dim = len(tab_cols)
    tab_dict = {bid: sub_norm.loc[bid].values.astype(np.float32) for bid in basin_ids if bid in sub_norm.index}
    return tab_dict, tab_dim, tab_cols


# ================== Prepare series ==================

def prepare_series_and_qstd(basin_ids: List[str], static_all: pd.DataFrame, cfg: Config):
    train_start = pd.to_datetime(cfg.TRAIN_START)
    train_end   = pd.to_datetime(cfg.TRAIN_END)
    val_start   = pd.to_datetime(cfg.VAL_START)
    val_end     = pd.to_datetime(cfg.VAL_END)

    static_sub = static_all.loc[basin_ids].copy()
    if "area_gages2" not in static_sub.columns:
        raise RuntimeError("STATIC_VARS must include area_gages2")

    series_train: Dict[str, pd.DataFrame] = {}
    series_val: Dict[str, pd.DataFrame] = {}
    q_std_by_basin: Dict[str, float] = {}

    print("Reading forcings & streamflow for all basins ...")
    for bid in tqdm(basin_ids, desc="CAMELS IO"):
        area_km2 = float(static_sub.loc[bid, "area_gages2"])
        try:
            q_series = read_streamflow(bid, area_km2, cfg.STREAMFLOW_DIR)
            forc = read_forcing(bid, cfg.FORCING_DIR, cfg.FORCING_VARS)
        except FileNotFoundError:
            continue

        df = align_forcing_flow(forc, q_series)
        if df.empty:
            continue

        mask_time = (df.index >= (val_start - pd.Timedelta(days=cfg.SEQ_LEN - 1))) & (df.index <= train_end)
        df = df.loc[mask_time].copy()
        if df.empty:
            continue

        df_train_target = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
        df_val_target   = df.loc[(df.index >= val_start) & (df.index <= val_end)].copy()

        df_train_full = df.loc[(df.index >= (train_start - pd.Timedelta(days=cfg.SEQ_LEN - 1))) & (df.index <= train_end)].copy()
        df_val_full   = df.loc[(df.index >= (val_start - pd.Timedelta(days=cfg.SEQ_LEN - 1))) & (df.index <= val_end)].copy()

        has_train = len(df_train_target) >= cfg.SEQ_LEN
        has_val   = len(df_val_target)   >= cfg.SEQ_LEN

        if has_train:
            series_train[bid] = df_train_full
            q_std_by_basin[bid] = float(df_train_target["q_mm"].std(ddof=0))
            if has_val:
                series_val[bid] = df_val_full

    if len(series_train) == 0:
        raise RuntimeError("No basin has enough training data with given SEQ_LEN.")
    return series_train, series_val, q_std_by_basin


# ================== Build z_batch + collect MID channel weights ==================

def build_z_batch(
    model: LSTMForecastStatic,
    basin_ids_batch: List[str],
    device: str,
    cfg: Config,
    static_low: Dict[str, np.ndarray],
    static_mid: Dict[str, np.ndarray],
    static_high: Dict[str, np.ndarray],
    static_tab: Dict[str, np.ndarray],
    tab_dim: int,
    C: int,
    return_mid_channel_w: bool = False,
) -> Tuple[torch.Tensor, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns:
      z_batch: (B, D_static)
      mid_channel_pack (optional): (uniq_ids[str], w_channel[U,C] float32 numpy)
    """
    b_np = np.array([str(b) for b in basin_ids_batch], dtype=object)
    uniq, inv = np.unique(b_np, return_inverse=True)
    U = len(uniq)

    z_parts_uniq = []
    w_mid_out = None

    # TAB
    if cfg.USE_TAB:
        tab_np = np.stack([static_tab[bid] for bid in uniq], axis=0).astype(np.float32)  # (U, tab_dim)
        tab_t = torch.from_numpy(tab_np).to(device, non_blocking=True)
        z_parts_uniq.append(tab_t)

    # LOW: mean only (U,C)
    if cfg.USE_LOW:
        low_np = np.stack([static_low[bid] for bid in uniq], axis=0).astype(np.float32)
        low_t = torch.from_numpy(low_np).to(device, non_blocking=True)
        z_parts_uniq.append(low_t)

    # MID: (U,C,M,M) -> (U,emb) + channel weights (U,C) if enabled
    if cfg.USE_MID:
        mid_np = np.stack([static_mid[bid] for bid in uniq], axis=0).astype(np.float32)
        mid_t = torch.from_numpy(mid_np).to(device, non_blocking=True)
        z_mid, w_mid = model.encode_mid(mid_t)  # w_mid: (U,C) or None
        z_parts_uniq.append(z_mid)

        if return_mid_channel_w and (w_mid is not None):
            w_mid_out = (uniq.copy(), w_mid.detach().cpu().numpy().astype(np.float32))

    # HIGH: bucket by K (variable K is supported)
    if cfg.USE_HIGH:
        k_list = [int(static_high[bid].shape[0]) for bid in uniq]
        buckets: Dict[int, List[int]] = {}
        for ui, K in enumerate(k_list):
            buckets.setdefault(K, []).append(ui)

        z_high_uniq = torch.zeros((U, cfg.CNN_EMB_DIM), device=device, dtype=torch.float32)

        for K, uidxs in buckets.items():
            high_np = np.stack([static_high[uniq[ui]] for ui in uidxs], axis=0).astype(np.float32)  # (U_k,K,C,P,P)
            high_t = torch.from_numpy(high_np).to(device, non_blocking=True)
            z_pool, _ = model.encode_high(high_t)  # (U_k, emb)
            z_high_uniq[torch.tensor(uidxs, device=device)] = z_pool

        z_parts_uniq.append(z_high_uniq)

    if len(z_parts_uniq) == 0:
        raise RuntimeError("All static switches are OFF. If you want pure dynamic LSTM, modify code to allow static_total_dim=0.")

    z_uniq = torch.cat(z_parts_uniq, dim=1)  # (U, D_static)
    inv_t = torch.from_numpy(inv).to(device)
    z_batch = z_uniq[inv_t]                  # (B, D_static)
    return z_batch, w_mid_out


# ================== Eval (with extra metrics) ==================

def eval_model(model: LSTMForecastStatic, loader: DataLoader, device: str,
               cfg: Config,
               static_low: Dict[str, np.ndarray],
               static_mid: Dict[str, np.ndarray],
               static_high: Dict[str, np.ndarray],
               static_tab: Dict[str, np.ndarray],
               tab_dim: int,
               C: int,
               horizon: int):
    model.eval()
    H = horizon
    basin_preds = [defaultdict(list) for _ in range(H)]
    basin_obs   = [defaultdict(list) for _ in range(H)]

    with torch.no_grad():
        for X_dyn, y_norm, basin_ids, q_stds in tqdm(loader, desc="Eval", leave=False):
            X_dyn = X_dyn.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)

            z_batch, _ = build_z_batch(
                model=model,
                basin_ids_batch=list(basin_ids),
                device=device,
                cfg=cfg,
                static_low=static_low,
                static_mid=static_mid,
                static_high=static_high,
                static_tab=static_tab,
                tab_dim=tab_dim,
                C=C,
                return_mid_channel_w=False,
            )

            y_hat_norm = model.forward_concat(X_dyn, z_batch)

            y_hat_q = denormalize_target_q(y_hat_norm.cpu().numpy())
            y_q     = denormalize_target_q(y_norm.cpu().numpy())

            B = y_hat_q.shape[0]
            for b in range(B):
                bid = str(basin_ids[b])
                for h in range(H):
                    basin_preds[h][bid].append(y_hat_q[b, h])
                    basin_obs[h][bid].append(y_q[b, h])

    mean_nse_list = []
    metrics_dfs = []
    for h in range(H):
        rows = []
        all_nse = []
        for bid in basin_preds[h].keys():
            sim = np.array(basin_preds[h][bid])
            obs = np.array(basin_obs[h][bid])
            met = {
                "basin_id": bid,
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

        df_metrics = pd.DataFrame(rows)
        metrics_dfs.append(df_metrics)
        mean_nse_list.append(float(np.nanmean(all_nse)) if len(all_nse) else np.nan)

    return mean_nse_list, metrics_dfs


# ================== Plot ==================

def plot_train_history(run_dir: Path, history: Dict[str, list], horizon: int):
    epochs = np.array(history["epoch"], dtype=int)
    train_loss = np.array(history["train_loss"], dtype=float)
    overall = np.array(history["val_mean_NSE_overall"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
    l1 = ax1.plot(epochs, train_loss, marker="o", label="Train Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    l2 = ax2.plot(epochs, overall, marker="s", label="Val Mean NSE", linewidth=1.5, color="tab:orange")
    ax2.set_ylabel("Val Mean NSE")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"Training History (H={horizon})")
    plt.tight_layout()
    out_fig = run_dir / "loss_curves.png"
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_fig}")


# ================== Main ==================

def train_and_eval():
    cfg = CFG
    set_seeds(cfg.SEED)

    device = cfg.DEVICE
    H = cfg.HORIZON

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.OUT_DIR / cfg.RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Outputs:", run_dir)
    print(f"Switches: TAB={cfg.USE_TAB}, LOW={cfg.USE_LOW}, MID={cfg.USE_MID}, HIGH={cfg.USE_HIGH}")
    print(f"Attn: MID_channel={cfg.USE_MID_CHANNEL_ATT}, HIGH_patch={cfg.USE_HIGH_PATCH_ATT}")
    print(f"MID={cfg.MID_SIZE}x{cfg.MID_SIZE}, HIGH patch P={cfg.HIGH_PATCH_SIZE}, stride={cfg.HIGH_STRIDE}, maxK={cfg.HIGH_MAX_PATCHES}")

    static_all = read_static_attributes(cfg.STATIC_DIR, cfg.STATIC_VARS)

    if cfg.USE_BASIN_CSV and cfg.BASIN_CSV.exists():
        df_ids = pd.read_csv(cfg.BASIN_CSV)
        basin_ids = df_ids["gauge_id"].astype(str).str.zfill(8).tolist()
    else:
        basin_ids = static_all.index.tolist()

    series_train, series_val, q_std_by_basin = prepare_series_and_qstd(basin_ids, static_all, cfg)
    basin_ids_all = sorted(set(series_train.keys()) | set(series_val.keys()))
    print(f"Train basins: {len(series_train)}, Val basins: {len(series_val)}, Union: {len(basin_ids_all)}")

    # TAB vectors
    if cfg.USE_TAB:
        static_tab, tab_dim, tab_cols = build_static_attr_vectors(static_all, basin_ids_all, cfg, run_dir)
    else:
        static_tab, tab_dim, tab_cols = {}, 0, []
        print("USE_TAB=False: no CAMELS static attributes will be used.")

    # static low/mid/high caches (HIGH now covering)
    static_low, static_mid, static_high, C, tif_names = build_static_low_mid_high(basin_ids_all, cfg)
    print(f"TIF channels (C={C}): {tif_names}")

    # filter basins that have required static pieces
    def ok(bid: str) -> bool:
        if cfg.USE_TAB and bid not in static_tab: return False
        if cfg.USE_LOW and bid not in static_low: return False
        if cfg.USE_MID and bid not in static_mid: return False
        if cfg.USE_HIGH and bid not in static_high: return False
        return True

    valid_ids = [bid for bid in basin_ids_all if ok(bid)]
    series_train = {bid: df for bid, df in series_train.items() if bid in valid_ids}
    series_val   = {bid: df for bid, df in series_val.items() if bid in valid_ids}
    q_std_by_basin = {bid: q_std_by_basin[bid] for bid in valid_ids if bid in q_std_by_basin}

    print(f"After static filter: Train basins={len(series_train)}, Val basins={len(series_val)}")

    # datasets/loaders
    train_ds = CamelsMultiStepDataset(
        series_by_basin=series_train,
        seq_len=cfg.SEQ_LEN,
        horizon=cfg.HORIZON,
        q_std_by_basin=q_std_by_basin,
        target_start=pd.to_datetime(cfg.TRAIN_START),
        target_end=pd.to_datetime(cfg.TRAIN_END),
        forcing_vars=cfg.FORCING_VARS,
    )
    val_ds = CamelsMultiStepDataset(
        series_by_basin=series_val,
        seq_len=cfg.SEQ_LEN,
        horizon=cfg.HORIZON,
        q_std_by_basin=q_std_by_basin,
        target_start=pd.to_datetime(cfg.VAL_START),
        target_end=pd.to_datetime(cfg.VAL_END),
        forcing_vars=cfg.FORCING_VARS,
    )

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

    # static_total_dim
    static_total_dim = 0
    if cfg.USE_TAB:  static_total_dim += tab_dim
    if cfg.USE_LOW:  static_total_dim += C
    if cfg.USE_MID:  static_total_dim += cfg.CNN_EMB_DIM
    if cfg.USE_HIGH: static_total_dim += cfg.CNN_EMB_DIM
    print(f"Static dims: TAB={tab_dim}, LOW(mean)={C}, MID={cfg.CNN_EMB_DIM}, HIGH={cfg.CNN_EMB_DIM} => total={static_total_dim}")

    model = LSTMForecastStatic(
        dyn_input_size=len(cfg.FORCING_VARS),
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        static_in_ch=C,
        cnn_emb_dim=cfg.CNN_EMB_DIM,
        static_total_dim=static_total_dim,
        horizon=cfg.HORIZON,
        cfg=cfg,
        initial_forget_bias=cfg.INITIAL_FORGET_BIAS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR_BASE)
    loss_fn = MultiStepNSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    history: Dict[str, list] = {"epoch": [], "train_loss": [], "val_mean_NSE_overall": []}
    for ld in range(H):
        history[f"val_mean_NSE_lead{ld}"] = []

    best_val_overall = -np.inf
    no_improve = 0

    # for channel weights output (MID-only)
    channel_weight_accum = defaultdict(list)  # basin_id -> list of (C,)
    save_channel_weights = cfg.USE_MID and cfg.USE_MID_CHANNEL_ATT

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for X_dyn, y_norm, basin_ids_batch, q_stds in pbar:
            X_dyn = X_dyn.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            q_stds = q_stds.to(device, non_blocking=True)

            z_batch, _ = build_z_batch(
                model=model,
                basin_ids_batch=list(basin_ids_batch),
                device=device,
                cfg=cfg,
                static_low=static_low,
                static_mid=static_mid,
                static_high=static_high,
                static_tab=static_tab,
                tab_dim=tab_dim,
                C=C,
                return_mid_channel_w=False,
            )

            optimizer.zero_grad(set_to_none=True)
            y_hat_norm = model.forward_concat(X_dyn, z_batch)
            loss = loss_fn(y_hat_norm, y_norm, q_stds)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(train_losses)))

        train_loss_mean = float(np.mean(train_losses))

        # collect channel weights on VAL once per epoch
        if save_channel_weights:
            model.eval()
            with torch.no_grad():
                for X_dyn, y_norm, basin_ids_b, q_stds in tqdm(val_loader, desc="Collect MID channel weights", leave=False):
                    _, pack = build_z_batch(
                        model=model,
                        basin_ids_batch=list(basin_ids_b),
                        device=device,
                        cfg=cfg,
                        static_low=static_low,
                        static_mid=static_mid,
                        static_high=static_high,
                        static_tab=static_tab,
                        tab_dim=tab_dim,
                        C=C,
                        return_mid_channel_w=True,
                    )
                    if pack is None:
                        continue
                    uniq_ids, w = pack  # w: (U,C)
                    for i, bid in enumerate(uniq_ids):
                        channel_weight_accum[bid].append(w[i])

        val_mean_nse_list, val_metrics_dfs = eval_model(
            model=model,
            loader=val_loader,
            device=device,
            cfg=cfg,
            static_low=static_low,
            static_mid=static_mid,
            static_high=static_high,
            static_tab=static_tab,
            tab_dim=tab_dim,
            C=C,
            horizon=cfg.HORIZON,
        )

        val_overall = float(np.nanmean(val_mean_nse_list))
        print(f"Epoch {epoch}: train_loss={train_loss_mean:.4f}, val_mean_NSE_overall={val_overall:.3f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_mean)
        history["val_mean_NSE_overall"].append(val_overall)
        for ld, v in enumerate(val_mean_nse_list):
            history[f"val_mean_NSE_lead{ld}"].append(float(v))

        scheduler.step(val_overall)

        # write metrics
        for ld, df_lead in enumerate(val_metrics_dfs):
            (run_dir / f"val_metrics_lead{ld}.csv").write_text(df_lead.to_csv(index=False))

        # early stop logic
        improved = val_overall > best_val_overall + 1e-4
        if improved:
            best_val_overall = val_overall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_overall": best_val_overall,
                    "config": cfg.__dict__,
                    "tif_names": tif_names,
                    "tab_cols": tab_cols,
                },
                run_dir / "camels_best.pt"
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val={best_val_overall:.3f}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # save history
    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
    plot_train_history(run_dir, history, horizon=cfg.HORIZON)

    # save basin-level channel weights
    if save_channel_weights and len(channel_weight_accum) > 0:
        rows = []
        for bid, ws in channel_weight_accum.items():
            wmean = np.mean(np.stack(ws, axis=0), axis=0)  # (C,)
            row = {"basin_id": bid}
            for j, name in enumerate(tif_names):
                row[f"w_{name}"] = float(wmean[j])
            rows.append(row)
        dfw = pd.DataFrame(rows).sort_values("basin_id")
        outp = run_dir / "channel_weights_by_basin_mid_only.csv"
        dfw.to_csv(outp, index=False)
        print(f"Saved MID-only channel weights: {outp}")

        gmean = dfw[[f"w_{n}" for n in tif_names]].mean(axis=0)
        df_global = pd.DataFrame([{"stat": "global_mean", **{k: float(v) for k, v in gmean.items()}}])
        df_global.to_csv(run_dir / "channel_weights_global_mean.csv", index=False)

    print("All done.")


if __name__ == "__main__":
    train_and_eval()
