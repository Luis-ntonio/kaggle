import torch
import torch.nn as nn
from typing import List, Tuple

class BFRBClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int       = 37,
        cnn_channels: List[int] = [64, 128],
        lstm_hidden: int     = 128,
        lstm_layers: int     = 2,
        dropout: float       = 0.6,
        num_bfrb_classes: int = 8,
        demo_dim: int        = 7
    ):
        """
        input_dim= 37  (sensores)
        demo_dim = número de features demográficas (p.ej. 7)
        """
        super().__init__()
        # --- CNN Backbone ---
        convs = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            convs += [
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2)
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*convs)

        # --- BiLSTM over CNN output ---
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # --- Shared MLP to get temporal embedding (128) ---
        self.shared_fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        # --- Fusion layer: concat(embed(128), demo_feat(demo_dim)) → 128 ---
        self.combined_fc = nn.Sequential(
            nn.Linear(128 + demo_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # --- Heads ---
        self.target_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.gesture_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_bfrb_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        length_mask: torch.Tensor,
        demo_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:          (B, Tmax, 37)
        length_mask:(B, Tmax) float mask
        demo_feat:  (B, demo_dim)

        Returns:
          logit_target: (B,)
          logit_gesture:(B, num_bfrb_classes)
        """
        B, Tmax, D = x.shape

        # 1) CNN → (B, C_last, Tcnn)
        x_c = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_c)

        # 2) Permute back → (B, Tcnn, C_last)
        cnn_out = cnn_out.permute(0, 2, 1)

        # 3) Recompute mask at Tcnn resolution
        n_pool = sum(1 for m in self.cnn if isinstance(m, nn.MaxPool1d))
        down = 2 ** n_pool
        new_len = (length_mask.sum(dim=1).long() // down).clamp(min=1)
        Tcnn = cnn_out.size(1)
        mask = torch.zeros(B, Tcnn, device=x.device)
        for i, nl in enumerate(new_len):
            mask[i, :nl] = 1.0

        # 4) BiLSTM + masked mean‐pool
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out * mask.unsqueeze(-1)
        summed = lstm_out.sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True) + 1e-6
        pooled = summed / counts  # (B, 2*lstm_hidden)

        # 5) Shared MLP → embed (B,128)
        embed = self.shared_fc(pooled)

        # 6) Fusion con demografía
        #    demo_feat debe venir de collate_fn como Tensor (B, demo_dim)
        combined = self.combined_fc(torch.cat([embed, demo_feat], dim=1))  # (B,128)

        # 7) Cabezas
        logit_target  = self.target_head(combined).squeeze(1)   # (B,)
        logit_gesture = self.gesture_head(combined)            # (B, num_bfrb_classes)

        return logit_target, logit_gesture
