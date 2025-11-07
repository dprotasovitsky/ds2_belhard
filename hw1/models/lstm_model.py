import torch
import torch.nn as nn


class AdvancedLSTMModel(nn.Module):
    """Расширенная LSTM модель с attention механизмом"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional=False,
        use_attention=False,
    ):
        super(AdvancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention механизм
        if use_attention:
            directional_multiplier = 2 if bidirectional else 1
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * directional_multiplier, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        # Полносвязные слои
        directional_multiplier = 2 if bidirectional else 1
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directional_multiplier, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # LSTM слой
        lstm_out, (hidden, cell) = self.lstm(x)

        # Применение attention или взятие последнего выхода
        if self.use_attention:
            attention_weights = torch.softmax(
                self.attention(lstm_out).squeeze(-1), dim=1
            )
            out = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        else:
            if self.bidirectional:
                out_forward = lstm_out[:, -1, : self.hidden_size]
                out_backward = lstm_out[:, -1, self.hidden_size :]
                out = torch.cat((out_forward, out_backward), dim=1)
            else:
                out = lstm_out[:, -1, :]

        # Полносвязные слои
        out = self.fc_layers(out)
        return out


class SimpleLSTMModel(nn.Module):
    """Простая LSTM модель для быстрого прототипирования"""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SimpleLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.linear(out)
        return out
