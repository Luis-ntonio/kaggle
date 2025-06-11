from BFRBClassifier import BFRBClassifier
from Dataset_Loader import BFRBDataset, collate_fn
from utils import compute_modality_stats
from tqdm import tqdm
# =============================================================================
# 1) Imports and global setup
# =============================================================================
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =============================================================================
# 5) Training / Validation Loop
# =============================================================================

def train_model(global_data: str, global_demo: str, train_csv: str, train_demo_csv: str,
                val_csv: str, val_demo_csv: str,
                device: str = "cuda", model = None):
    """
    Train the BFRBClassifier using the train split, evaluate on a held-out validation split.
    """

    # 1) Build Dataset (with no stats initially)
    global_ = BFRBDataset(global_data, global_demo, imu_stats=None, thm_stats=None, tof_stats=None, is_train=True, global_stats=True)
    #save map and idx for inference run
    map, idx, target, gesture, ids, indexes, _ = global_.get_lists()
    np.save("./out/bfrb_map.npy", np.array(map))
    np.save("./out/idx2bfrb.npy", np.array(idx))
    np.save("./out/target.npy", np.array(target))
    np.save("./out/gesture.npy", np.array(gesture))
    np.save("./out/ids.npy", np.array(ids))
    # Save indexes as a list of arrays using numpy's save with allow_pickle=True
    np.save("./out/indexes.npy", np.array(indexes, dtype=object), allow_pickle=True)


    # 2) Compute normalization stats on TRAINING set only

    #try loading stats from disk
    if Path("./out/imu_mean.npy").exists():
        imu_mean = np.load("./out/imu_mean.npy")
        imu_std = np.load("./out/imu_std.npy")
        thm_mean = np.load("./out/thm_mean.npy")
        thm_std = np.load("./out/thm_std.npy")
        tof_mean = np.load("./out/tof_mean.npy")
        tof_std = np.load("./out/tof_std.npy")
    else:
        train_ds = BFRBDataset(global_data, global_demo, imu_stats=None, thm_stats=None, tof_stats=None, is_train=True)
        (imu_mean, imu_std), (thm_mean, thm_std), (tof_mean, tof_std) = compute_modality_stats(train_ds, device=device)
        np.save("./out/imu_mean.npy", imu_mean)
        np.save("./out/imu_std.npy", imu_std)
        np.save("./out/thm_mean.npy", thm_mean)
        np.save("./out/thm_std.npy", thm_std)
        np.save("./out/tof_mean.npy", tof_mean)
        np.save("./out/tof_std.npy", tof_std)
    
    # 3) Rebuild Dataset with computed stats
    train_ds = BFRBDataset(global_data, global_demo,
                           imu_stats=(imu_mean, imu_std),
                           thm_stats=(thm_mean, thm_std),
                           tof_stats=(tof_mean, tof_std),
                           is_train=True, bfrb_map=map, idx2bfrb=idx, target= target, gesture=gesture, ids = ids, indexes = indexes)


    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 4) Instantiate model
    if model is None:
        model = BFRBClassifier(input_dim=37, num_bfrb_classes=len(map)).to(device)
    else:
        model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.3)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()  # we’ll mask out where is_target=0

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, 31)):
        model.train()
        train_losses = []
        for batch in train_loader:
            feats = batch["features"].to(device)         # (B, Tmax, 37)
            len_mask = batch["length_mask"].to(device)   # (B, Tmax)
            demo_feats = batch["demo_feat"].to(device)  # (B, demo_dim)
            is_targ = batch["is_target"].to(device)      # (B,)
            gest_id = batch["gesture_id"].to(device)     # (B,)

            optimizer.zero_grad()
            logits_targ, logits_gest = model(feats, len_mask, demo_feats)

            # 1) Binary loss
            L_target = bce_loss(logits_targ, is_targ)

            # 2) Multiclass loss (only for true targets)
            loss_g = ce_loss(logits_gest, gest_id)  # (B,)

            loss = L_target + loss_g
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())



        avg_train = np.mean(train_losses)
        print(f"Epoch {epoch}: TrainLoss={avg_train:.4f}")
        scheduler.step(avg_train)

        # Save best
        if avg_train < best_val_loss:
            best_val_loss = avg_train
            torch.save(model.state_dict(), "./out/best_bfrb_model.pt")
            print("  → saved new best model")

    print("Training complete. Best val loss:", best_val_loss)
 


# =============================================================================
# 7) Example usage
# =============================================================================
if __name__ == "__main__":
    # Paths (replace with your actual CSV paths)
    TRAIN_CSV = "./data/train.csv"
    TRAIN_DEMO = "./data/train_demographics.csv"
    TEST_CSV = "./data/test.csv"
    TEST_DEMO = "./data/test_demographics.csv"
    MODEL = "./out/best_bfrb_model.pt"
    # Load model if exists
    model = None
    if Path(MODEL).exists():
        print("Loading existing model from:", MODEL)
        model = BFRBClassifier(input_dim=37, num_bfrb_classes=18)  # Adjust num_bfrb_classes as needed
        model.load_state_dict(torch.load(MODEL))
    else:
        print("No existing model found, training a new one.")

    # 1) Read full train.csv into a Pandas DataFrame
    full_train_df = pd.read_csv(TRAIN_CSV)

    # 2) Split its unique sequence_id’s into train vs. val (80/20 split)
    all_seq_ids = full_train_df["sequence_id"].unique()
    train_seq_ids, val_seq_ids = train_test_split(
        all_seq_ids, test_size=0.20, random_state=SEED, shuffle=True
    )

    # 3) Create two new DataFrames: one for train‐split, one for val‐split
    train_split_df = full_train_df[ full_train_df["sequence_id"].isin(train_seq_ids) ].copy()
    val_split_df   = full_train_df[ full_train_df["sequence_id"].isin(val_seq_ids)   ].copy()

    # 4) Write them out to disk so BFRBDataset can read them by path
    train_split_df.to_csv("train_split.csv", index=False)
    val_split_df.to_csv("val_split.csv",     index=False)

    # We do *not* need a separate val_demographics.csv, because demographics are never used
    # to assign labels inside BFRBDataset. Passing the same TRAIN_DEMO twice is fine.
    VAL_DEMO = TRAIN_DEMO


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Train model (compute stats + train)
    imu_stats, thm_stats, tof_stats = None, None, None
    train_model(
        TRAIN_CSV, TRAIN_DEMO,
        "train_split.csv", TRAIN_DEMO,
        "val_split.csv", VAL_DEMO,
        device=device,
        model=model
    )

    # save imu, thm, tof stats for later use

