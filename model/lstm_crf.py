import torch
import torch.autograd as autograd
import torch.nn as nn
import model.crf as crf
import model.utils as utils


class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, large_CRF=True):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio)
        self.rnn_layers = rnn_layers
    
        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.tagset_size = tagset_size
        if large_CRF:
            self.crf = crf.CRF_L(hidden_dim, tagset_size)
        else:
            self.crf = crf.CRF_S(hidden_dim, tagset_size)

        self.batch_size = 1
        self.seq_length = 1

    def rand_init_hidden(self):
        return autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)), autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_batch_size(self, bsize):
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def rand_init_embedding(self):
        utils.init_embedding(self.word_embeds.weight)

    def rand_init(self, init_embedding=False):
        if init_embedding:
            utils.init_embedding(self.word_embeds.weight)
        utils.init_lstm(self.lstm)
        self.crf.rand_init()

    def forward(self, sentence, hidden=None):
        self.set_batch_seq_size(sentence)

        embeds = self.word_embeds(sentence)
        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden)
        lstm_out = lstm_out.view(-1, self.hidden_dim)

        d_lstm_out = self.dropout2(lstm_out)

        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.seq_length, self.batch_size, self.tagset_size, self.tagset_size)
        return crf_out, hidden