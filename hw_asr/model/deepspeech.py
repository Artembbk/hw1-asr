from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence

from hw_asr.base import BaseModel


class DeepSpeech2pacModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        dim = 128  # Пример значения dim

        # Размеры ядер сверток и шаги
        self.kernel_size1 = (41, 11)
        self.stride1 = (2, 2)
        self.kernel_size2 = (21, 11)
        self.stride2 = (2, 1)
        self.kernel_size3 = (21, 11)
        self.stride3 = (2, 1)
        self.channels1 = 32
        self.channels2 = 32
        self.channels3 = 96
        self.fc_hidden = fc_hidden
        self.n_class = n_class

        self.out_dim1 = ((dim - self.kernel_size1[0]) // self.stride1[0]) + 1
        self.out_dim2 = ((self.out_dim1 - self.kernel_size2[0]) // self.stride2[0]) + 1
        self.out_dim3 = ((self.out_dim2 - self.kernel_size3[0]) // self.stride3[0]) + 1

        self.conv1 = nn.Conv2d(1, self.channels1, self.kernel_size1, stride=self.stride1)
        self.conv2 = nn.Conv2d(self.channels1, self.channels2, self.kernel_size2, stride=self.stride2)
        self.conv3 = nn.Conv2d(self.channels2, self.channels3, self.kernel_size3, stride=self.stride3)

        self.batchnorm1 = nn.BatchNorm2d(self.channels1)
        self.batchnorm2 = nn.BatchNorm2d(self.channels2)
        self.batchnorm3 = nn.BatchNorm2d(self.channels3)

        self.relu = nn.ReLU()

        self.rnn = nn.GRU(self.out_dim3 * self.channels3, fc_hidden, num_layers=1, batch_first=True)

        self.fc = nn.Linear(in_features=fc_hidden, out_features=n_class)



    def forward(self, spectrogram, **batch):
        assert len(spectrogram.size()) == 3
        batch_size, dim, time = spectrogram.size()
        assert batch_size == len(batch['audio_path'])
        output = spectrogram.unsqueeze(1)
        output = self.relu(self.conv1(output))
        assert output.size()[2] == self.out_dim1
        output = self.relu(self.conv2(output))
        assert output.size()[2] == self.out_dim2
        output = self.relu(self.conv3(output))
        assert output.size()[2] == self.out_dim3

        batch_size, channels, dim, seq_l = output.size()
        assert channels == self.channels3
        assert batch_size == len(batch['audio_path'])
        output = output.permute(0, 3, 1, 2)
        output = output.view(batch_size, seq_l, channels * dim)

        output = self.rnn(output)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.fc_hidden
        output = self.fc(output)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.n_class

        return output


    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
