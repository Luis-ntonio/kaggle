import os
import polars as pl
import torch
import numpy as np
import xgboost as xgb
from torch.utils.data import DataLoader
from Dataset_Loader import BFRBDataset, collate_fn
from BFRB_Xgboost import FeatureExtractorCNN   # o el módulo donde lo tengas
from utils import preprocess_sequence_df

# 1) Carga CNN + XGBoost al inicio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = FeatureExtractorCNN(input_dim=37, cnn_channels=[64,128], dropout=0.3)
#cnn_model.load_state_dict(torch.load("./out/cnn_feature_extractor.pt", map_location=device))
cnn_model.to(device).eval()

bst = xgb.Booster()
bst.load_model("./out/bfrb_xgboost_model.json")

# Mapea índice XGB → nombre de gesto
bfrb_map = np.load("./out/bfrb_map.npy", allow_pickle=True).item()
inv_map  = {v:k for k,v in bfrb_map.items()}

# Estadísticas para el preprocess
imu_mean = np.load("./out/imu_mean.npy")
imu_std  = np.load("./out/imu_std.npy")
thm_mean = np.load("./out/thm_mean.npy")
thm_std  = np.load("./out/thm_std.npy")
tof_mean = np.load("./out/tof_mean.npy")
tof_std  = np.load("./out/tof_std.npy")

def predict(test_csv: pl.DataFrame, test_demo_csv: pl.DataFrame) -> str:
    # 2) Montar DataFrame completo con demografía
    df_seq = test_csv.to_pandas()
    df_demo= test_demo_csv.to_pandas()
    total = df_seq.merge(df_demo, on="subject", how="left")

    # 3) Preprocess sensors
    X_seq, len_mask = preprocess_sequence_df(
        total,
        imu_mean, imu_std,
        thm_mean, thm_std,
        tof_mean, tof_std
    )
    # demografía
    demo_row = total.iloc[0][[
        "adult_child","age","sex","handedness",
        "height_cm","shoulder_to_wrist_cm","elbow_to_wrist_cm"
    ]].values.astype(np.float32)

    # 4) Pasar por CNN feature‐extractor
    with torch.no_grad():
        x_tensor = X_seq.unsqueeze(0).to(device)            # (1, T, 37)
        mask_t   = len_mask.unsqueeze(0).to(device)         # (1, T)
        # our CNN extractor espera (x, mask) y produce (1, C)
        feat_cnn = cnn_model(x_tensor, mask_t)              # (1, C)
    # 5) Concatenar demografía
    feat = torch.cat([feat_cnn.cpu(), torch.tensor(demo_row).unsqueeze(0)], dim=1).numpy()  
    #    -> shape (1, C+7)

    # 6) XGBoost predict
    dmat = xgb.DMatrix(feat)
    probs= bst.predict(dmat)        # (1, num_classes)
    pred = int(np.argmax(probs, axis=1)[0])

    # 7) Buscar en inv_map y devolver la cadena
    gesture_name = inv_map[pred]
    return gesture_name

# Conectar con CMIInferenceServer
import kaggle_evaluation.cmi_inference_server as cmi_server
inference_server = cmi_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "val_split.csv",
            "./data/train_demographics.csv"
        )
    )
