import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import model.crf as crf
import model.utils as utils
import model.highway as highway

class LM_LSTM_CRF(nn.Module):
    def __init__(self, tagset_size, char_size, char_dim, char_hidden_dim, char_rnn_layers, embedding_dim, word_hidden_dim, word_rnn_layers, vocab_size, dropout_ratio, large_CRF=True, if_highway = False, in_doc_words = 2, highway_layers = 1):

        super(LM_LSTM_CRF, self).__init__()
        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_size = char_size
        self.word_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_size = vocab_size
        self.if_highway = if_highway

        self.char_embeds = nn.Embedding(char_size, char_dim)
        self.forw_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers=char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.back_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers=char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.char_rnn_layers = char_rnn_layers

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.word_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)

        self.word_rnn_layers = word_rnn_layers

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.tagset_size = tagset_size
        if large_CRF:
            self.crf = crf.CRF_L(word_hidden_dim, tagset_size)
        else:
            self.crf = crf.CRF_S(word_hidden_dim, tagset_size)

        if if_highway:
            self.forw2char = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.back2char = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.forw2word = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.back2word = highway.hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.fb2char = highway.hw(2 * char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)

        self.char_pre_train_out = nn.Linear(char_hidden_dim, char_size)
        self.word_pre_train_out = nn.Linear(char_hidden_dim, in_doc_words)

        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_size(self, bsize):
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def rand_init_embedding(self):
        utils.init_embedding(self.char_embeds.weight)

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        assert (pre_word_embeddings.size()[1] == self.word_dim)
        self.word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self, init_char_embedding=True, init_word_embedding=False):
        if init_char_embedding:
            utils.init_embedding(self.char_embeds.weight)
        if init_word_embedding:
            utils.init_embedding(self.word_embeds.weight)
        if self.if_highway:
            self.forw2char.rand_init()
            self.back2char.rand_init()
            self.forw2word.rand_init()
            self.back2word.rand_init()
            self.fb2char.rand_init()
        utils.init_lstm(self.forw_char_lstm)
        utils.init_lstm(self.back_char_lstm)
        utils.init_lstm(self.word_lstm)
        utils.init_linear(self.char_pre_train_out)
        utils.init_linear(self.word_pre_train_out)
        self.crf.rand_init()

    def char_pre_train_forward(self, sentence, hidden=None):
        #sentence: seq_len_char * batch
        #original order
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.forw_char_lstm(d_embeds)
        lstm_out = lstm_out.view(-1, self.char_hidden_dim)
        d_lstm_out = self.dropout(lstm_out)
        if self.if_highway:
            char_out = self.forw2char(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out
        pre_score = self.char_pre_train_out(d_char_out)
        return pre_score, hidden

    def char_pre_train_backward(self, sentence, hidden=None):
        #sentence: seq_len_char * batch
        #reverse order
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.back_char_lstm(d_embeds)
        lstm_out = lstm_out.view(-1, self.char_hidden_dim)
        d_lstm_out = self.dropout(lstm_out)
        if self.if_highway:
            char_out = self.forw2char(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out
        pre_score = self.char_pre_train_out(d_char_out)
        return pre_score, hidden

    def word_pre_train_forward(self, sentence, position, hidden=None):
        #sentence: seq_len_char * batch
        #original order
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.forw_char_lstm(d_embeds)

        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1], self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.char_hidden_dim)

        if self.if_highway:
            char_out = self.forw2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out

        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def word_pre_train_backward(self, sentence, position, hidden=None):
        #sentence: seq_len_char * batch
        #reverse order
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.back_char_lstm(d_embeds)
        
        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1], self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.char_hidden_dim)

        if self.if_highway:
            char_out = self.back2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out

        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def forward(self, forw_sentence, forw_position, back_sentence, back_position, word_seq, hidden=None):
        #forw_sentence: seq_len_char * batch
        #forw_position: seq_len_word * batch
        #back_sentence: seq_len_char * batch
        #back_position: seq_len_word * batch
        #word_seq: seq_len_word * batch

        self.set_batch_seq_size(forw_position)

        #embedding layer
        forw_emb = self.char_embeds(forw_sentence)
        back_emb = self.char_embeds(back_sentence)

        #dropout
        d_f_emb = self.dropout(forw_emb)
        d_b_emb = self.dropout(back_emb)

        #forward the whole sequence
        forw_lstm_out, _ = self.forw_char_lstm(d_f_emb)#seq_len_char * batch * char_hidden_dim

        back_lstm_out, _ = self.back_char_lstm(d_b_emb)#seq_len_char * batch * char_hidden_dim

        #select predict point
        forw_position = forw_position.unsqueeze(2).expand(self.word_seq_length, self.batch_size, self.char_hidden_dim)
        select_forw_lstm_out = torch.gather(forw_lstm_out, 0, forw_position)

        back_position = back_position.unsqueeze(2).expand(self.word_seq_length, self.batch_size, self.char_hidden_dim)
        select_back_lstm_out = torch.gather(back_lstm_out, 0, back_position)

        fb_lstm_out = self.dropout(torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2))
        if self.if_highway:
            char_out = self.fb2char(fb_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = fb_lstm_out

        #word
        word_emb = self.word_embeds(word_seq)
        d_word_emb = self.dropout(word_emb)

        #combine
        word_input = torch.cat((d_word_emb, d_char_out), dim = 2)

        #word level lstm
        lstm_out, _ = self.word_lstm(word_input)
        d_lstm_out = self.dropout(lstm_out)

        #convert to crf
        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.word_seq_length, self.batch_size, self.tagset_size, self.tagset_size)
        
        return crf_out