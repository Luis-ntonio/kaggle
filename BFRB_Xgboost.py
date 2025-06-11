import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from local_metric import score
from Dataset_Loader import BFRBDataset, collate_fn

# -----------------------------------------------------------------------------
# 1) Define un bloque de Attention Pooling sobre la dimensión temporal
# -----------------------------------------------------------------------------
class AttnPool1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, T, C), mask: (B, T) con 1.0 en pasos válidos
        scores = self.attn(x).squeeze(-1)                       # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)           # descartar padding
        alpha  = F.softmax(scores, dim=1).unsqueeze(-1)        # (B, T, 1)
        return (x * alpha).sum(dim=1)                          # (B, C)

# -----------------------------------------------------------------------------
# 2) FeatureExtractorCNN revisado: CNN + SEBlock opcional + AttnPool + concat demografía
# -----------------------------------------------------------------------------
class FeatureExtractorCNN(nn.Module):
    def __init__(self, input_dim=37, cnn_channels=[64,128], dropout=0.3):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.attnpool = AttnPool1D(cnn_channels[-1])

    def forward(self, x: torch.Tensor, length_mask: torch.Tensor):
        # x: (B, Tmax, 37), length_mask: (B, Tmax)
        B, Tmax, _ = x.shape
        x_c = x.permute(0,2,1)            # (B, 37, Tmax)
        cnn_out = self.cnn(x_c)           # (B, C, Tcnn)
        Tcnn = cnn_out.size(2)
        # rebuild mask at Tcnn resolution
        factor = Tmax // Tcnn
        mask_down = F.max_pool1d(length_mask.unsqueeze(1), kernel_size=factor).squeeze(1)
        # permute para AttnPool
        cnn_out_t = cnn_out.permute(0,2,1) # (B, Tcnn, C)
        pooled   = self.attnpool(cnn_out_t, mask_down)  # (B, C)
        return pooled

# -----------------------------------------------------------------------------
# 3) Modifica tu Dataset & collate_fn para exponer demo_feat
# -----------------------------------------------------------------------------
# En __getitem__ de BFRBDataset añade tras preprocess_sequence_df:
#    demo_row = df_seq.iloc[0][["adult_child","age","sex","handedness",
#                               "height_cm","shoulder_to_wrist_cm","elbow_to_wrist_cm"]].values
#    output["demo_feat"] = torch.tensor(demo_row, dtype=torch.float32)
#
# Y en collate_fn incluye:
#    demo_feats = torch.stack([b["demo_feat"] for b in batch], dim=0)  # (B,7)
#    and return it under "demo_feat"

# -----------------------------------------------------------------------------
# 4) Extrae features CNN + demografía
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = FeatureExtractorCNN(input_dim=37).to(device)

    # dataset y loader
    train_ds = BFRBDataset("./data/train.csv", "./data/train_demographics.csv", is_train=True)
    bfrb_map   = train_ds.bfrb_map
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

def extract_features(model, loader, device):
    model.eval()
    feats_list = []
    labs_list  = []
    with torch.no_grad():
        for batch in loader:
            x     = batch["features"].to(device)
            mask  = batch["length_mask"].to(device)
            demo  = batch["demo_feat"].to(device)
            labs  = batch["gesture_id"].numpy()
            # cnn features
            f_cnn = model(x, mask)             # (B, C)
            # concat demografía
            f_cat = torch.cat([f_cnn, demo], dim=1)  # (B, C+7)
            feats_list.append(f_cat.cpu().numpy())
            labs_list.append(labs)
    X = np.vstack(feats_list)
    y = np.concatenate(labs_list)
    return X, y

if __name__ == "__main__":

    X, y = extract_features(cnn_model, train_loader, device)

# -----------------------------------------------------------------------------
# 5) Entrena XGBoost con búsqueda de hiperparámetros simple y evaluación jerárquica
# -----------------------------------------------------------------------------
def train_xgboost(X, y):
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval   = xgb.DMatrix(Xval, label=yval)

    params = {
        "objective":   "multi:softprob",
        "num_class":   len(bfrb_map),
        "max_depth":   12,
        "eta":         0.05,
        "subsample":   0.6,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss"
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dval,"val")],
        verbose_eval=20
    )

    # Predicción y construcción de DataFrames para la métrica jerárquica
    yprob = bst.predict(dval)
    ypred = np.argmax(yprob, axis=1)

    # Mapeo inverso índice→gesture
    inv_map = {v:k for k,v in bfrb_map.items()}

    df_pred = pd.DataFrame({
        "sequence_id": [train_ds.seq_ids[i] for i in range(len(ypred))],
        "gesture":     [inv_map[c]   for c in ypred]
    })
    df_true = pd.DataFrame({
        "sequence_id": [train_ds.seq_ids[i] for i in range(len(yval))],
        "gesture":     [inv_map[c]   for c in yval]
    })

    f1 = score(df_true, df_pred, row_id_column_name="sequence_id")
    print("Hierarchical F1 en validación:", f1)
    return bst


if __name__ == "__main__":
    bst = train_xgboost(X, y)

    # Save the trained XGBoost model
    bst.save_model("./out/bfrb_xgboost_model.json")

