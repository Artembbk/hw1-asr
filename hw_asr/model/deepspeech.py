from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence
import torch

from hw_asr.base import BaseModel

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Веса для обновления состояния (Update Gate)
        self.W_z = nn.Linear(hidden_size, hidden_size)
        self.U_z = nn.Linear(input_size, hidden_size)
        
        # Веса для создания нового состояния (New State)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.U_r = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.U_h = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x, h):
        # Расчет вектора обновления (Update Gate)
        z = torch.sigmoid(self.W_z(h) + self.bn(self.U_z(x)))
        
        # Расчет вектора сброса (Reset Gate)
        r = torch.sigmoid(self.W_r(h) + self.bn(self.U_r(x)))
        
        # Расчет нового состояния (New State)
        new_state = torch.tanh(self.W_h(r * h) + self.bn(self.U_h(x)))
        
        # Обновление текущего состояния (Hidden State Update)
        h_new = (1 - z) * h + z * new_state
        
        return h_new
    
import torch
import torch.nn as nn

class CustomGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True):
        super(CustomGRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1  # 2 for bidirectional, 1 for unidirectional
        self.gru_cells = nn.ModuleList([CustomGRU(input_size, hidden_size) for _ in range(self.directions * num_layers)])

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(self.directions * self.num_layers, batch_size, self.hidden_size)
        # If bidirectional, we need to account for the forward and backward directions.
        if self.bidirectional:
            self.directions = 2

        # Get the number of time steps and batch size
        seq_len, batch_size, _ = x.size()

        # List to store outputs
        output = []

        for layer in range(self.num_layers):
            layer_outputs = []
            for direction in range(self.directions):
                h_direction = h[direction + layer * self.directions]

                for t in range(seq_len):
                    h_direction = self.gru_cells[direction + layer * self.directions](x[t], h_direction)
                    layer_outputs.append(h_direction)

            # Update the hidden state for the next layer
            h = torch.stack(layer_outputs).view(self.directions * self.num_layers, seq_len, batch_size, self.hidden_size)

        # Return outputs and the final hidden state
        output = torch.stack(layer_outputs).view(seq_len, batch_size, self.directions, self.hidden_size)
        return output, h


class DeepSpeech2pacModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden, conv_num, conv_type, 
                 kernel_sizes, strides, channels, rnn_type, 
                 rnn_layers, rnn_bidirectional, batch_norm_conv, **batch):
        super().__init__(n_feats, n_class, **batch)

        rnns = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN,
            'bn_gru': CustomGRULayer
        }

        self.rnn_bidirectional = rnn_bidirectional
        self.kernel_sizes = kernel_sizes
        self.conv_num = conv_num
        self.strides = strides
        self.channels = [1] + channels
        self.hidden = hidden
        self.n_class = n_class
        self.n_feats = n_feats

        self.out_conv_dim = n_feats
        for i in range(conv_num):
            self.out_conv_dim = ((self.out_conv_dim - self.kernel_sizes[i][0]) // self.strides[i][0]) + 1

        self.conv_layers = []

        for i in range(conv_num):
            if conv_type == '2d':
                self.conv_layers.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], stride=self.strides[i]))
            else:
                raise NotImplementedError
            
            if batch_norm_conv:
                self.conv_layers.append(nn.BatchNorm2d(self.channels[i+1]))

            self.conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*self.conv_layers)

        rnn = rnns[rnn_type]
        self.rnn = rnn(self.out_conv_dim * self.channels[-1], self.hidden, num_layers=rnn_layers, bidirectional=rnn_bidirectional, batch_first=True)
        if rnn_bidirectional:
            self.fc = nn.Linear(in_features=2*hidden, out_features=n_class)
        else:
            self.fc = nn.Linear(in_features=hidden, out_features=n_class)



    def forward(self, spectrogram, **batch):
        assert len(spectrogram.size()) == 3
        batch_size, dim, time = spectrogram.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.n_feats
        output = spectrogram.unsqueeze(1)

        output = self.conv_layers(output)

        batch_size, channels, dim, seq_l = output.size()
        assert channels == self.channels[-1]
        assert batch_size == len(batch['audio_path'])
        assert dim == self.out_conv_dim
        output = output.permute(0, 3, 1, 2)
        output = output.view(batch_size, seq_l, channels * dim)

        output, _ = self.rnn(output)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        if self.rnn_bidirectional:
            assert dim == 2*self.hidden
        else:
            assert dim == self.hidden
        output = self.fc(output)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.n_class
        return {"logits": output}


    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for i in range(self.conv_num):
            output_lengths = ((output_lengths - self.kernel_sizes[i][1]) // self.strides[i][1]) + 1
        return output_lengths
