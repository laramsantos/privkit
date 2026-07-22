
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, p=0.3):
        super(Encoder, self).__init__()
        self.in_drop = nn.Dropout(p)
        self.fc = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=p)

    def forward(self, x):
        x = self.in_drop(x)
        x = torch.relu(self.fc(x))
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, p=0.3):
        super(Decoder, self).__init__()
        self.in_drop = nn.Dropout(p)
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=p)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, steps, teacher_ratio=0.3, target_sequence=None):
        predictions = []
        prev_point = x[:, -1, :]

        use_teacher = (torch.rand((), device=x.device) < teacher_ratio) if self.training else False

        for i in range(steps):
            input_step = prev_point if i == 0 else (target_sequence[:, i-1, :] if use_teacher else prev_point)
            # add dropout on the decoder input path
            input_step = self.in_drop(input_step)
            input_step = torch.relu(self.fc_in(input_step)).unsqueeze(1)
            output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
            predicted_position = self.fc_out(output.squeeze(1))
            predictions.append(predicted_position)
            prev_point = predicted_position

        return torch.stack(predictions, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, target_sequence=None, teacher_forcing_ratio=0.3):
        hidden, cell = self.encoder(x)
        steps = target_sequence.shape[1] if target_sequence is not None else 5
        return self.decoder(x, hidden, cell, steps, teacher_forcing_ratio, target_sequence)
