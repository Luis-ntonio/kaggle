import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader
# =============================================================================
# 2) Utility functions for modality-specific preprocessing
# =============================================================================

def compute_tof_statistics(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a single DataFrame row, compute per-ToF-sensor (mean, max, min) 
    ignoring -1 (no signal). Also return a mask array of shape (5,) 
    =1 if sensor had any valid pixel, else=0.
    """
    means = []
    maxs = []
    mins = []
    mask = []
    for s in range(1, 6):  # sensors 1..5
        pixels = row[[f"tof_{s}_v{i}" for i in range(64)]].values.astype(np.float32)
        # Identify valid pixels (pix >= 0)
        valid = pixels >= 0
        if valid.sum() == 0:
            # no valid return → mask=0, stats=0
            mask.append(0)
            means.append(0.0)
            maxs.append(0.0)
            mins.append(0.0)
        else:
            mask.append(1)
            valid_pixels = pixels[valid]
            means.append(valid_pixels.mean())
            maxs.append(valid_pixels.max())
            mins.append(valid_pixels.min())
    return np.array(means + maxs + mins, dtype=np.float32), np.array(mask, dtype=np.float32)


def preprocess_sequence_df(df_seq: pd.DataFrame,
                           imu_mean, imu_std,
                           thm_mean, thm_std,
                           tof_agg_mean, tof_agg_std) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    df_seq: a Pandas DataFrame slice for one sequence_id (all T rows).
    imu_mean, imu_std: either (np.ndarray of shape (7,), np.ndarray of shape (7,)) or (None, None)
    thm_mean, thm_std: either (np.ndarray(5,), np.ndarray(5,)) or (None, None)
    tof_agg_mean, tof_agg_std: either (np.ndarray(15,), np.ndarray(15,)) or (None, None)

    Returns:
      X: torch.FloatTensor (T, 37)
      length_mask: torch.FloatTensor (T,) all ones
    """

    rows = []
    for _, row in df_seq.iterrows():
        # 1) IMU: acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z → 7 dims
        imu_vals = row[["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]].values.astype(np.float32)
        imu_vals = np.nan_to_num(imu_vals, nan=0.0)
        # ONLY normalize if stats are provided
        if (imu_mean is not None) and (imu_std is not None):
            imu_vals = (imu_vals - imu_mean) / (imu_std + 1e-6)

        # 2) Thermopile: thm_1..thm_5 → 5 dims + mask
        thm    = row[["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]].values.astype(np.float32)
        thm_mask = (~np.isnan(thm)).astype(np.float32)   # 1 if real sensor reading, 0 if NaN
        thm = np.nan_to_num(thm, nan=0.0)
        if (thm_mean is not None) and (thm_std is not None):
            thm = (thm - thm_mean) / (thm_std + 1e-6)

        # 3) ToF aggregates: (mean, max, min) over each 8×8 → 15 dims + 5‐dim mask
        tof_stats, tof_mask = compute_tof_statistics(row)  # returns np.array(15,), np.array(5,)
        tof_stats = np.nan_to_num(tof_stats, nan=0.0, posinf=0.0, neginf=0.0)
        if (tof_agg_mean is not None) and (tof_agg_std is not None):
            tof_stats = (tof_stats - tof_agg_mean) / (tof_agg_std + 1e-6)

        # 4) Concatenate features into a 37‐dim vector
        feat = np.concatenate([
            imu_vals,        # 7 dims
            thm,             # 5 dims
            thm_mask,        # 5 dims
            tof_stats,       # 15 dims
            tof_mask         # 5 dims
        ]).astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(feat)

    X = np.stack(rows, axis=0)  # shape (T, 37)
    length_mask = np.ones((X.shape[0],), dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(length_mask)


def compute_modality_stats(dataset, device="cpu") -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
    """
    Compute global mean/std for:
      - IMU channels (7 dims)
      - Thermopile channels (5 dims)
      - ToF aggregated channels (15 dims)

    We'll iterate over all sequences, accumulate sums and sums of squares, then divide.
    IMPORTANT: We ignore NaNs and treat them as zero after masking.

    Returns:
      (imu_mean, imu_std), (thm_mean, thm_std), (tof_agg_mean, tof_agg_std)
      where each “mean” & “std” is a NumPy array on CPU.
    """
    from Dataset_Loader import collate_fn  # adjust this import path as needed

    # 1) Allocate accumulators on the chosen device
    imu_sum      = torch.zeros(7,  device=device)
    imu_sq_sum   = torch.zeros(7,  device=device)

    thm_sum      = torch.zeros(5,  device=device)
    thm_sq_sum   = torch.zeros(5,  device=device)

    tof_sum      = torch.zeros(15, device=device)
    tof_sq_sum   = torch.zeros(15, device=device)

    # We will keep counts as Torch tensors on the same device, to avoid mixing CPU/GPU.
    # For IMU, it’s a single scalar count; for the other two, per‐channel counts.
    count_imu = torch.tensor(0.0,      device=device)  # scalar
    count_thm = torch.zeros(5, device=device)          # one count per thermopile channel
    count_tof = torch.zeros(15, device=device)         # one count per aggregated ToF channel

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    for batch in loader:
        feats     = batch["features"].to(device)      # (B, Tmax, 37)
        len_mask  = batch["length_mask"].to(device)   # (B, Tmax)
        B, Tmax, _ = feats.shape

        # Extract sub‐tensors: IMU = feats[:, :, 0:7], THM = feats[:, :, 7:12], THM_mask = feats[:, :, 12:17],
        # ToF_stats = feats[:, :, 17:32], ToF_mask = feats[:, :, 32:37]

        # 1) IMU block: reshape to (B*Tmax, 7)
        imu_block = feats[:, :, 0:7].reshape(-1, 7)  # (B*Tmax, 7)
        # All IMU rows are “valid” (we imputed NaN→0), so each row counts
        imu_sum    += imu_block.sum(dim=0)           # (7,)
        imu_sq_sum += (imu_block * imu_block).sum(dim=0)
        count_imu  += float(B * Tmax)                # scalar

        # 2) Thermopile block: raw values are feats[:, :, 7:12]; mask is feats[:, :, 12:17]
        thm_block = feats[:, :, 7:12].reshape(-1, 5)       # (B*Tmax, 5)
        thm_mask  = feats[:, :, 12:17].reshape(-1, 5)      # (B*Tmax, 5) of 0/1

        # Sum only where mask=1
        thm_sum    += (thm_block * thm_mask).sum(dim=0)             # (5,)
        thm_sq_sum += ((thm_block * thm_mask) ** 2).sum(dim=0)      # (5,)
        count_thm  += thm_mask.sum(dim=0)                           # (5,) on device

        # 3) ToF‐aggregated block: feats[:, :, 17:32] are the 15 dims of (mean,max,min)
        #    ToF_mask is feats[:, :, 32:37] (one per each of the 5 sensors)
        tof_stats = feats[:, :, 17:32].reshape(-1, 15)   # (B*Tmax, 15)
        tof_mask  = feats[:, :, 32:37].reshape(-1, 5)    # (B*Tmax, 5)

        # We need a mask of shape (B*Tmax, 15). Each group of 3 dims in “tof_stats” shares the same
        # 1/0 mask from the corresponding sensor (5 sensors). E.g. dims [0..2] correspond to sensor1,
        # dims [3..5] to sensor2, ... dims [12..14] to sensor5.
        # We can replicate each of the 5 mask columns three times:
        tiled_tof_mask = torch.cat([tof_mask, tof_mask, tof_mask], dim=1)  # (B*Tmax, 15)

        tof_sum    += (tof_stats * tiled_tof_mask).sum(dim=0)              # (15,)
        tof_sq_sum += ((tof_stats * tiled_tof_mask) ** 2).sum(dim=0)       # (15,)
        count_tof  += tiled_tof_mask.sum(dim=0)                            # (15,)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Now compute means & standard deviations (all still on `device`). Then
    #    move to CPU and convert to NumPy arrays **only once** at the very end.
    # ─────────────────────────────────────────────────────────────────────────────

    # 4a) IMU mean & var (scalar count_imu)
    imu_mean_t = imu_sum / (count_imu + 1e-6)                   # (7,) tensor on device
    imu_var_t  = (imu_sq_sum / (count_imu + 1e-6)) - (imu_mean_t ** 2)
    imu_var_t  = torch.clamp(imu_var_t, min=1e-6)
    imu_std_t  = torch.sqrt(imu_var_t)

    # 4b) THM mean & var (per-channel)
    thm_mean_t = thm_sum / (count_thm + 1e-6)                   # (5,) tensor on device
    thm_var_t  = (thm_sq_sum / (count_thm + 1e-6)) - (thm_mean_t ** 2)
    thm_var_t  = torch.clamp(thm_var_t, min=1e-6)
    thm_std_t  = torch.sqrt(thm_var_t)

    # 4c) ToF mean & var (15 dims)
    tof_mean_t = tof_sum / (count_tof + 1e-6)                   # (15,) tensor on device
    tof_var_t  = (tof_sq_sum / (count_tof + 1e-6)) - (tof_mean_t ** 2)
    tof_var_t  = torch.clamp(tof_var_t, min=1e-6)
    tof_std_t  = torch.sqrt(tof_var_t)

    # 5) Move everything to CPU and convert to NumPy arrays
    imu_mean     = imu_mean_t.cpu().numpy()      # shape (7,)
    imu_std      = imu_std_t.cpu().numpy()       # shape (7,)

    thm_mean     = thm_mean_t.cpu().numpy()      # shape (5,)
    thm_std      = thm_std_t.cpu().numpy()       # shape (5,)

    tof_agg_mean = tof_mean_t.cpu().numpy()      # shape (15,)
    tof_agg_std  = tof_std_t.cpu().numpy()       # shape (15,)

    return (imu_mean, imu_std), (thm_mean, thm_std), (tof_agg_mean, tof_agg_std)