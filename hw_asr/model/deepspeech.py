from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence

from hw_asr.base import BaseModel


class DeepSpeech2pacModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden, conv_num, conv_type, 
                 kernel_sizes, strides, channels, rnn_type, 
                 rnn_layers, rnn_bidirectional, batch_norm_conv, **batch):
        super().__init__(n_feats, n_class, **batch)

        rnns = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN,
        }

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
                self.conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            self.conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*self.conv_layers)

        rnn = rnns[rnn_type]
        self.rnn = rnn(self.out_conv_dim * self.channels[-1], self.hidden, num_layers=rnn_layers, bidirectional=rnn_bidirectional, batch_first=True)

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
        assert dim == self.hidden
        output = self.fc(output)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.n_class
        return output


    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for i in range(self.conv_num):
            output_lengths = ((output_lengths - self.kernel_sizes[i][1]) // self.strides[i][1]) + 1

        return output_lengths
