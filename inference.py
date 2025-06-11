import os
import polars as pl
import torch
import numpy as np
#from Improve_BFRB import ImprovedBFRBClassifier as BFRBClassifier
from BFRBClassifier import BFRBClassifier
from torch.utils.data import DataLoader
from utils import preprocess_sequence_df
import csv
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#test2
"""model = BFRBClassifier(
        input_dim=37,
        demog_dim=7,
        inception_channels=[64,128],    # u otros que prefieras
        transform_dim=256,
        transformer_heads=8,
        transformer_layers=3,
        drop=0.3,
        num_classes=18
    ).to(device)"""
#test1
model = BFRBClassifier(input_dim=37, num_bfrb_classes=18).to(device)
model.load_state_dict(torch.load("./out/best_bfrb_model.pt", map_location=device))
model.eval()

imu_mean = np.load("./out/imu_mean.npy")    # (7,)
imu_std  = np.load("./out/imu_std.npy")     # (7,)

thm_mean = np.load("./out/thm_mean.npy")    # (5,)
thm_std  = np.load("./out/thm_std.npy")     # (5,)

tof_mean = np.load("./out/tof_mean.npy")    # (15,)
tof_std  = np.load("./out/tof_std.npy")     # (15,)

brfb_map = np.load("./out/bfrb_map.npy", allow_pickle=True).item()  # {0: "BFRB_0", 1: "BFRB_1", ...}

# Save arrays to CSV files
# Save all arrays into a single CSV, each as a column
arrays_dict = {
    "imu_mean": imu_mean,
    "imu_std": imu_std,
    "thm_mean": thm_mean,
    "thm_std": thm_std,
    "tof_mean": tof_mean,
    "tof_std": tof_std
}

# Find the maximum length among all arrays
max_len = max(len(arr) for arr in arrays_dict.values())

# Pad arrays with NaN to have the same length
for k in arrays_dict:
    arr = arrays_dict[k]
    if len(arr) < max_len:
        arrays_dict[k] = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

df = pd.DataFrame(arrays_dict)
df.to_csv("./out/all_means_stds.csv", index=False)

# Save brfb_map as a two-column CSV
brfb_df = pd.DataFrame(list(brfb_map.items()), columns=["key", "value"])
brfb_df.to_csv("./out/bfrb_map.csv", index=False)



# —————————————————————————————————————————————————————————————————————————————————————————————————————
# Ahora reescribimos predict para que reciba un *único* bloque “test_csv” (Polars DF de N filas de la secuencia)
# y retorne sólo la etiqueta final como string.
# —————————————————————————————————————————————————————————————————————————————————————————————————————

def predict(test_csv: pl.DataFrame, test_demo_csv: pl.DataFrame) -> str:
    """
    Cada llamada corresponde EXACTAMENTE a una secuencia de test (todas las filas para el mismo sequence_id).
    Debemos devolver **solo** la clase predicha, convertida a str: "-1" si no es BFRB, o "0".."7" si es alguna BFRB.
    
    Parámetros:
      - test_csv:       Polars DataFrame con todas las filas (T time steps) para una sola sequence_id.
      - test_demo_csv:  Polars DataFrame de demographics (un solo renglón), que no usamos acá.
    Retorno:
      - Una cadena, p.ej. "-1" o "3".
    """

    # 1) Convertir la secuencia a Pandas para usar preprocess_sequence_df tal cual:
    df_seq_pd = test_csv.to_pandas()
    df_demo_pd = test_demo_csv.to_pandas()

    total_df = df_seq_pd.merge(
        df_demo_pd,
        on="subject",
        how="left"
    )

    # 2) Preprocesar fila‐por‐fila en un solo batch:
    #    (devuelve X: Tensor (T,37), len_mask: Tensor (T,))
    X, len_mask = preprocess_sequence_df(
        total_df,
        imu_mean        =   imu_mean, 
        imu_std         =   imu_std,
        thm_mean        =   thm_mean, 
        thm_std         =   thm_std,
        tof_agg_mean    =   tof_mean, 
        tof_agg_std     =   tof_std
    )

    #select only "adult_child", "age", "sex", "handedness", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm" from demo_df
    demo_row = total_df.iloc[0][["adult_child", "age", "sex",
                                   "handedness", "height_cm",
                                   "shoulder_to_wrist_cm",
                                   "elbow_to_wrist_cm"]].values.astype(np.float32)
    
    demo_feat = torch.tensor(demo_row, dtype=torch.float32).to(device)  # shape (7,)

    # 3) Añadir dimensión de batch (1, T, 37) y mover a device
    X = X.unsqueeze(0).to(device)          # (1, T, 37)
    #para test1
    #len_mask = len_mask.unsqueeze(0).to(device)  # (1, T)
    demo_feat = demo_feat.unsqueeze(0).to(device)  # (1, 7)
    padding_mask = (len_mask == 0.0).unsqueeze(0).to(device)  # (1, 1, T)
    # 4) Forward en el modelo
    with torch.no_grad():
        logit_targ, logit_gest = model(X, padding_mask, demo_feat)
        clase = int(torch.argmax(logit_gest, dim=1).item())

    # 5) Devolver solo la etiqueta como string
    for key, clase_name in brfb_map.items():
        if clase == clase_name:
            print(f"Predicción: {key}")
            return str(key)
            

# —————————————————————————————————————————————————————————————————————————————————————————————————————
# Igual que antes, conectamos CMIInferenceServer:
# —————————————————————————————————————————————————————————————————————————————————————————————————————
import kaggle_evaluation.cmi_inference_server as cmi_server

inference_server = cmi_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "./data/test.csv",
            "./data/test_demographics.csv"
        )
    )
