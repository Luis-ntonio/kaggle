import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple
from utils import preprocess_sequence_df  # Asegúrate de que sea la versión que maneja stats=None
from torch.nn.utils.rnn import pad_sequence

# =============================================================================
# 3) PyTorch Dataset & DataLoader
# =============================================================================
class BFRBDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 demo_path: str,
                 imu_stats: Tuple[np.ndarray, np.ndarray] = None,
                 thm_stats: Tuple[np.ndarray, np.ndarray] = None,
                 tof_stats: Tuple[np.ndarray, np.ndarray] = None,
                 is_train: bool = True, global_stats: bool = False, 
                 bfrb_map: map = None, idx2bfrb =None, target=None, 
                 gesture=None, ids=None, indexes=None, tof_percentiles=None):
        """
        csv_path: ruta a train.csv o test.csv
        demo_path: ruta a train_demographics.csv o test_demographics.csv
        imu_stats, thm_stats, tof_stats: tuplas (mean, std) de cada bloque de sensores
        is_train: si True, se esperan columnas 'gesture' y 'sequence_type' para calcular etiquetas.
        """

        super().__init__()
        print(f"Loading BFRB dataset from {csv_path} and {demo_path}, is_train={is_train}")
        
        # 1) Cargamos los DataFrames (siempre convertimos a Pandas internamente)
        if csv_path and demo_path:
            self.df = pd.read_csv(csv_path)
            self.demo = pd.read_csv(demo_path)
            self.df = self.df.merge(
                self.demo,
                on="subject",
                how="left"
            )
            print(self.df.shape, self.demo.shape)
        else:
            # Permite instanciar con csv_path=None en modo predicción
            self.df = pd.DataFrame()
            self.demo = pd.DataFrame()
        

        self.is_train = is_train

        # 2) Guardamos las estadísticas de normalización (o None)
        self.imu_mean, self.imu_std     = imu_stats if imu_stats is not None else (None, None)
        self.thm_mean, self.thm_std     = thm_stats if thm_stats is not None else (None, None)
        self.tof_agg_mean, self.tof_agg_std = tof_stats if tof_stats is not None else (None, None)

        # 3) Preparar diccionario de mapeo gesto→índice  (solo si is_train=True)
        #    Además, preparamos idx2bfrb (índice→nombre de gesto) para poder reconstruirlo después.
        self.bfrb_map = bfrb_map if bfrb_map else {}
        self.idx2bfrb = idx2bfrb if idx2bfrb else []
        self.tof_percentiles = tof_percentiles if tof_percentiles else {}
        self.is_target_list = []
        self.gesture_id_list = []
        self.seq_ids = []
        self.group_indices = []

        if self.is_train and bfrb_map is None:
            # Asegurémonos de comparar en minúsculas, en caso de inconsistencias
            self.df["sequence_type_lower"] = self.df["sequence_type"].str.lower()
            
            # Filtramos únicamente las filas que son “target” (BFRB)
            gest_list = (
                self.df["gesture"]
                .unique()
                .tolist()
            )
            gest_list = sorted(gest_list)  # Lista ordenada de nombres de gesto
            for idx, g in enumerate(gest_list):
                self.bfrb_map[g] = idx
                self.idx2bfrb.append(g)

        self.unique_seq_ids = self.df["sequence_id"].unique().tolist()

            # Ahora: len(self.idx2bfrb) debe ser 8
        # 4) Agrupamos filas por sequence_id → obtenemos self.seq_ids y self.group_indices

        if not self.df.empty:
            last_seq = None
            current_indices = []
            for idx, seq in enumerate(self.df["sequence_id"].values):
                if seq != last_seq and current_indices:
                    self.seq_ids.append(last_seq)
                    self.group_indices.append(np.array(current_indices, dtype=np.int64))
                    current_indices = []
                current_indices.append(idx)
                last_seq = seq
            # Finalizamos la última secuencia
            if current_indices:
                self.seq_ids.append(last_seq)
                self.group_indices.append(np.array(current_indices, dtype=np.int64))

            # 5) Si estamos en entrenamiento, construimos is_target_list y gesture_id_list
            if self.is_train:
            # Debe corresponder longitud de seq_ids == longitud de group_indices
                for seq_idx_arr in self.group_indices:
                    r = seq_idx_arr[0]  # tomo la primera fila de esa secuencia
                    seq_type = str(self.df.loc[r, "sequence_type"]).lower()
                    gname = self.df.loc[r, "gesture"]
                    self.gesture_id_list.append(self.bfrb_map[gname])
                    if seq_type == "target":
                        # Clase BFRB: asigno is_target=1 y obtengo el índice en bfrb_map
                        self.is_target_list.append(1)
                        if gname not in self.bfrb_map:
                            raise KeyError(f"Gesto '{gname}' no estaba en bfrb_map.")
                    else:
                        # Non‐target: is_target=0 y gesto = -1
                        self.is_target_list.append(0)
                        
            print("Warning: The DataFrame is empty. No sequences to process.")
        print(f"Total sequences: {len(self.seq_ids)}")
    
    def get_lists(self):
        return self.bfrb_map, self.idx2bfrb, self.is_target_list, self.gesture_id_list, self.seq_ids, self.group_indices, self.unique_seq_ids
        
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx: int):
        """
        Devuelve un diccionario con:
        - 'sequence_id': int                                <--- añadido
        - 'features':     torch.Tensor (T, 37)
        - 'length_mask':  torch.Tensor (T,)
        - 'is_target':    int (0/1)     [solo si is_train=True]
        - 'gesture_id':   int (0..7 o -1)[solo si is_train=True]
        """
        seq_id = self.seq_ids[idx]
        indices = self.group_indices[idx]
        df_seq = self.df.iloc[indices].reset_index(drop=True)

        # Preprocesamos las filas de esta secuencia → (X, length_mask)
        X, len_mask = preprocess_sequence_df(
            df_seq,
            imu_mean=self.imu_mean,
            imu_std=self.imu_std,
            thm_mean=self.thm_mean,
            thm_std=self.thm_std,
            tof_agg_mean=self.tof_agg_mean,
            tof_agg_std=self.tof_agg_std,
            tof_percentiles=self.tof_percentiles  # Asegúrate de pasar los percentiles aquí
        )

        demo_row = df_seq.iloc[0][["adult_child", "age", "sex",
                                "handedness", "height_cm",
                                "shoulder_to_wrist_cm",
                                "elbow_to_wrist_cm"]].values.astype(np.float32)

        demo_feat = torch.tensor(demo_row, dtype=torch.float32)  # shape (7,)

        output = {
            "features": X,  # Tensor (T, 37)
            "length_mask": len_mask,  # Tensor (T,)
            "demo_feat": demo_feat  # (7,)
        }

        if self.is_train:
            output["sequence_id"] = seq_id  # sequence_id is text, keep as string
            output["is_target"] = torch.tensor(self.is_target_list[idx], dtype=torch.float32)
            output["gesture_id"] = torch.tensor(self.gesture_id_list[idx], dtype=torch.long)
        return output



def collate_fn(batch: List[dict]):
    """
    Pad cada 'features' al máximo T en el batch.
    Retorna:
      - features_padded:    FloatTensor (B, Tmax, 37)
      - length_masks_padded:FloatTensor (B, Tmax)
      - sequence_id:        LongTensor (B,)
      - is_target:          FloatTensor (B,) or None
      - gesture_id:         LongTensor (B,) or None
      - seq_lengths:        LongTensor (B,) = longitudes originales
    """
    # 1) Sacamos features, masks y sequence_ids
    feats = [b["features"]     for b in batch]
    masks = [b["length_mask"]  for b in batch]
    demo_feats = torch.stack([b["demo_feat"] for b in batch], dim=0)  # (B,7)
    seq_ids = [b["sequence_id"] for b in batch]  # secuencia_id como texto

    # 2) Padding
    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)  # (B, Tmax, 37)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0.0)  # (B, Tmax)

    padding_mask = (masks_padded == 0.0)  # BoolTensor (B, Tmax)

    out = {
        "features":    feats_padded,
        'padding_mask': padding_mask,  # (B, Tmax)
        "length_mask": masks_padded,
        "demo_feat":   demo_feats,
        "seq_lengths": torch.tensor([f.shape[0] for f in feats], dtype=torch.long),
        "sequence_id": seq_ids  # Convertimos a LongTensor
    }
    if "is_target" in batch[0]:
        out["is_target"]  = torch.stack([b["is_target"] for b in batch])
        out["gesture_id"] = torch.stack([b["gesture_id"] for b in batch])
    return out
