import torch
from torch import nn as nn

class ModelBiLSTM_v2(nn.Module):
    def __init__(self,
                 kmer : int = 21,
                 hidden_size : int = 256,
                 embed_size : list = [16, 4],
                 dropout_rate : float = 0.5,
                 num_layer1 : int = 2,
                 num_layer2 : int = 3,
                 num_classes : int = 2,
                 ):
        super(ModelBiLSTM_v2, self).__init__()
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(embed_size[0], embed_size[1])
        self.lstm_seq = nn.LSTM(8,
                                hidden_size // 2,
                                num_layers=num_layer1,
                                batch_first=True,
                                dropout=dropout_rate,
                                bidirectional=True,
                                )

        self.linear_seq = nn.Linear(hidden_size, hidden_size // 2)

        self.lstm_signal = nn.LSTM(15,
                                   (hidden_size // 2),
                                   num_layers=num_layer1,
                                   batch_first=True,
                                   dropout=dropout_rate,
                                   bidirectional=True,
                                   )
        self.linear_signal = nn.Linear(hidden_size, hidden_size // 2)

        self.lstm_comb = nn.LSTM(hidden_size,
                                 hidden_size,
                                 num_layers=num_layer2,
                                 batch_first=True,
                                 dropout=dropout_rate,
                                 bidirectional=True)

        self.drop_out = nn.Dropout(p=dropout_rate)

        self.linear_out_1 = nn.Linear(hidden_size * 2, 2)
        self.linear_out_2 = nn.Linear(2 * kmer, num_classes)

        self.soft_max = nn.Softmax(1)

    def forward(self, kmer, signals):
        kmer_embed = self.embed(kmer.long())
        # signals = signals.reshape(signals.shape[0], signals.shape[2], signals.shape[3])

        out_seq = torch.cat((kmer_embed, signals[:, :, :4]), 2)

        out_signal = signals[:, :, 4:]
        out_seq, _ = self.lstm_seq(out_seq)  # (N, L, nhid_seq * 2)
        out_seq = self.linear_seq(out_seq)  # (N, L, nhid_seq)
        out_seq = self.relu(out_seq)

        out_signal, _ = self.lstm_signal(out_signal)
        out_signal = self.linear_signal(out_signal)  # (N, L, nhid_signal)
        out_signal = self.relu(out_signal)

        # combined ================================================
        out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(out, )  # (N, L, hidden_size * 2)

        out = self.drop_out(out)
        out = self.linear_out_1(out).flatten(1)
        out = self.drop_out(out)
        out = self.linear_out_2(out)

        return out, self.soft_max(out)