import torch
import torch.nn as nn
from typing import List, Tuple

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T)
        attn_weights = self.attn(x).squeeze(-1)  # (B, T)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, T)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled

class BFRBClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int       = 40,
        cnn_channels: List[int] = [64, 128],
        rnn_hidden: int     = 128,
        dropout: float       = 0.5,
        num_bfrb_classes: int = 8,
        demo_dim: int        = 7
    ):
        super().__init__()
        # CNN Backbone
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

        # BiGRU and BiLSTM in parallel
        self.bigru = nn.GRU(input_size=cnn_channels[-1], hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.bilstm = nn.LSTM(input_size=cnn_channels[-1], hidden_size=rnn_hidden, batch_first=True, bidirectional=True)

        # Attention pooling
        self.attn_pool = AttentionPooling(input_dim=4 * rnn_hidden)

        # Shared MLP
        self.shared_fc = nn.Sequential(
            nn.Linear(4 * rnn_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        # Combine with demo features
        self.combined_fc = nn.Sequential(
            nn.Linear(128 + demo_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Output heads
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

    def forward(self, x: torch.Tensor, length_mask: torch.Tensor, demo_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Tmax, D = x.shape
        x_c = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_c).permute(0, 2, 1)  # (B, T', C)

        # Downsample mask
        n_pool = sum(1 for m in self.cnn if isinstance(m, nn.MaxPool1d))
        down = 2 ** n_pool
        new_len = (length_mask.sum(dim=1).long() // down).clamp(min=1)
        Tcnn = cnn_out.size(1)
        mask = torch.zeros(B, Tcnn, device=x.device)
        for i, nl in enumerate(new_len):
            mask[i, :nl] = 1.0

        # RNN paths
        gru_out, _ = self.bigru(cnn_out)
        lstm_out, _ = self.bilstm(cnn_out)
        rnn_combined = torch.cat([gru_out, lstm_out], dim=-1)  # (B, T', 4*rnn_hidden)

        pooled = self.attn_pool(rnn_combined, mask)
        embed = self.shared_fc(pooled)
        combined = self.combined_fc(torch.cat([embed, demo_feat], dim=1))

        logit_target = self.target_head(combined).squeeze(1)
        logit_gesture = self.gesture_head(combined)
        return logit_target, logit_gesture
