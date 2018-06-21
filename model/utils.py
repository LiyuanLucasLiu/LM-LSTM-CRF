"""
.. module:: utils
    :synopsis: utility tools

.. moduleauthor:: Liyuan Liu, Frank Xu
"""

import codecs
import csv
import itertools
from functools import reduce

import numpy as np
import shutil
import torch
import json

import torch.nn as nn
import torch.nn.init

from model.ner_dataset import *

zip = getattr(itertools, 'izip', zip)


def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """helper function to calculate argmax of input vector at dimension 1
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum

    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def switch(vec1, vec2, mask):
    """
    switch function for pytorch

    args:
        vec1 (any size) : input tensor corresponding to 0
        vec2 (same to vec1) : input tensor corresponding to 1
        mask (same to vec1) : input tensor, each element equals to 0/1
    return:
        vec (*)
    """
    catvec = torch.cat([vec1.view(-1, 1), vec2.view(-1, 1)], dim=1)
    switched_vec = torch.gather(catvec, 1, mask.long().view(-1, 1))
    return switched_vec.view(-1)


def encode2char_safe(input_lines, char_dict):
    """
    get char representation of lines

    args:
        input_lines (list of strings) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def concatChar(input_lines, char_dict):
    """
    concat char into string

    args:
        input_lines (list of list of char) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    features = [[char_dict[' ']] + list(reduce(lambda x, y: x + [char_dict[' ']] + y, sentence)) + [char_dict['\n']] for sentence in input_lines]
    return features


def encode_safe(input_lines, word_dict, unk):
    """
    encode list of strings into word-level representation with unk
    """
    lines = list(map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines


def encode(input_lines, word_dict):
    """
    encode list of strings into word-level representation
    """
    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines


def encode2Tensor(input_lines, word_dict, unk):
    """
    encode list of strings into word-level representation (tensor) with unk
    """
    lines = list(map(lambda t: torch.LongTensor(list(map(lambda m: word_dict.get(m, unk), t))), input_lines))
    return lines


def generate_corpus_char(lines, if_shrink_c_feature=False, c_thresholds=1, if_shrink_w_feature=False, w_thresholds=1):
    """
    generate label, feature, word dictionary, char dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_c_feature: whether shrink char-dictionary
        c_threshold: threshold for shrinking char-dictionary
        if_shrink_w_feature: whether shrink word-dictionary
        w_threshold: threshold for shrinking word-dictionary

    """
    features, labels, feature_map, label_map = generate_corpus(lines, if_shrink_feature=if_shrink_w_feature, thresholds=w_thresholds)
    char_count = dict()
    for feature in features:
        for word in feature:
            for tup in word:
                if tup not in char_count:
                    char_count[tup] = 0
                else:
                    char_count[tup] += 1
    if if_shrink_c_feature:
        shrink_char_count = [k for (k, v) in iter(char_count.items()) if v >= c_thresholds]
        char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}
    else:
        char_map = {k: v for (v, k) in enumerate(char_count.keys())}
    char_map['<u>'] = len(char_map)  # unk for char
    char_map[' '] = len(char_map)  # concat for char
    char_map['\n'] = len(char_map)  # eof for char
    return features, labels, feature_map, label_map, char_map

def shrink_features(feature_map, features, thresholds):
    """
    filter un-common features by threshold
    """
    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (ind + 1) for ind in range(0, len(shrinked_feature_count))}

    #inserting unk to be 0 encoded
    feature_map['<unk>'] = 0
    #inserting eof
    feature_map['<eof>'] = len(feature_map)
    return feature_map

def generate_corpus(lines, if_shrink_feature=False, thresholds=1):
    """
    generate label, feature, word dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_feature: whether shrink word-dictionary
        threshold: threshold for shrinking word-dictionary

    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    feature_map = dict()
    label_map = dict()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if line[0] not in feature_map:
                feature_map[line[0]] = len(feature_map) + 1 #0 is for unk
            tmp_ll.append(line[-1])
            if line[-1] not in label_map:
                label_map[line[-1]] = len(label_map)
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)
    label_map['<start>'] = len(label_map)
    label_map['<pad>'] = len(label_map)
    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        #inserting unk to be 0 encoded
        feature_map['<unk>'] = 0
        #inserting eof
        feature_map['<eof>'] = len(feature_map)

    return features, labels, feature_map, label_map


def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels

def read_features(lines, multi_docs = True):
    """
    convert un-annotated corpus into features
    """
    if multi_docs:
        documents = list()
        features = list()
        tmp_fl = list()
        for line in lines:
            if_doc_end = (len(line) > 10 and line[0:10] == '-DOCSTART-')
            if not (line.isspace() or if_doc_end):
                line = line.split()[0]
                tmp_fl.append(line)
            else:
                if len(tmp_fl) > 0:
                    features.append(tmp_fl)
                    tmp_fl = list()
                if if_doc_end and len(features) > 0:
                    documents.append(features)
                    features = list()
        if len(tmp_fl) > 0:
            features.append(tmp_fl)
        if len(features) >0:
            documents.append(features)
        return documents
    else:
        features = list()
        tmp_fl = list()
        for line in lines:
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.split()[0]
                tmp_fl.append(line)
            elif len(tmp_fl) > 0:
                features.append(tmp_fl)
                tmp_fl = list()
        if len(tmp_fl) > 0:
            features.append(tmp_fl)

        return features

def shrink_embedding(feature_map, word_dict, word_embedding, caseless):
    """
    shrink embedding dictionary to in-doc words only
    """
    if caseless:
        feature_map = set([k.lower() for k in feature_map.keys()])
    new_word_list = [k for k in word_dict.keys() if (k in feature_map)]
    new_word_dict = {k:v for (v, k) in enumerate(new_word_list)}
    new_word_list_ind = torch.LongTensor([word_dict[k] for k in new_word_list])
    new_embedding = word_embedding[new_word_list_ind]
    return new_word_dict, new_embedding

def encode_corpus(lines, f_map, l_map, if_lower = False):
    """
    encode corpus into features and labels
    """
    tmp_fl = []
    tmp_ll = []
    features = []
    labels = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)
    if if_lower:
        features = list(map(lambda t: list(map(lambda x: x.lower(), t)), features))
    feature_e = encode_safe(features, f_map, f_map['<unk>'])
    label_e = encode(labels, l_map)
    return feature_e, label_e


def encode_corpus_c(lines, f_map, l_map, c_map):
    """
    encode corpus into features (both word-level and char-level) and labels
    """
    tmp_fl = []
    tmp_ll = []
    features = []
    labels = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    feature_c = encode2char_safe(features, c_map)
    feature_e = encode_safe(features, f_map, f_map['<unk>'])
    label_e = encode(labels, l_map)
    return feature_c, feature_e, label_e

def load_embedding(emb_file, delimiter, feature_map, caseless, unk, shrink_to_train=False):
    """
    load embedding
    """
    if caseless:
        feature_set = set([key.lower() for key in feature_map])
    else:
        feature_set = set([key for key in feature_map])

    word_dict = dict()
    embedding_array = list()
    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        if shrink_to_train and line[0] not in feature_set:
            continue
        if line[0] == unk:
            word_dict['<unk>'] = len(word_dict)
        else:
            word_dict[line[0]] = len(word_dict)
        embedding_array.append(vector)
    embedding_tensor_1 = torch.FloatTensor(np.asarray(embedding_array))
    emb_len = embedding_tensor_1.size(1)

    rand_embedding_count = 0
    for key in feature_map:
        if caseless:
            key = key.lower()
        if key not in word_dict:
            word_dict[key] = len(word_dict)
            rand_embedding_count += 1

    rand_embedding_tensor = torch.FloatTensor(rand_embedding_count, emb_len)
    init_embedding(rand_embedding_tensor)

    embedding_tensor = torch.cat((embedding_tensor_1, rand_embedding_tensor), 0)
    return word_dict, embedding_tensor

def load_embedding_wlm(emb_file, delimiter, feature_map, full_feature_set, caseless, unk, emb_len, shrink_to_train=False, shrink_to_corpus=False):
    """
    load embedding, indoc words would be listed before outdoc words

    args:
        emb_file: path to embedding file
        delimiter: delimiter of lines
        feature_map: word dictionary
        full_feature_set: all words in the corpus
        caseless: convert into casesless style
        unk: string for unknown token
        emb_len: dimension of embedding vectors
        shrink_to_train: whether to shrink out-of-training set or not
        shrink_to_corpus: whether to shrink out-of-corpus or not
    """
    if caseless:
        feature_set = set([key.lower() for key in feature_map])
        full_feature_set = set([key.lower() for key in full_feature_set])
    else:
        feature_set = set([key for key in feature_map])
        full_feature_set = set([key for key in full_feature_set])

    #ensure <unk> is 0
    word_dict = {v:(k+1) for (k,v) in enumerate(feature_set - set(['<unk>']))}
    word_dict['<unk>'] = 0

    in_doc_freq_num = len(word_dict)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, emb_len)
    init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = list()
    outdoc_embedding_array = list()
    outdoc_word_array = list()

    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if shrink_to_train and line[0] not in feature_set:
            continue

        if line[0] == unk:
            rand_embedding_tensor[0] = torch.FloatTensor(vector) #unk is 0
        elif line[0] in word_dict:
            rand_embedding_tensor[word_dict[line[0]]] = torch.FloatTensor(vector)
        elif line[0] in full_feature_set:
            indoc_embedding_array.append(vector)
            indoc_word_array.append(line[0])
        elif not shrink_to_corpus:
            outdoc_word_array.append(line[0])
            outdoc_embedding_array.append(vector)

    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))

    if not shrink_to_corpus:
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))
        word_emb_len = embedding_tensor_0.size(1)
        assert(word_emb_len == emb_len)

    if shrink_to_corpus:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
    else:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)

    for word in indoc_word_array:
        word_dict[word] = len(word_dict)
    in_doc_num = len(word_dict)
    if  not shrink_to_corpus:
        for word in outdoc_word_array:
            word_dict[word] = len(word_dict)

    return word_dict, embedding_tensor, in_doc_num

def calc_threshold_mean(features):
    """
    calculate the threshold for bucket by mean
    """
    lines_len = list(map(lambda t: len(t) + 1, features))
    average = int(sum(lines_len) / len(lines_len))
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t >= average, lines_len))
    lower_average = int(sum(lower_line) / len(lower_line))
    upper_average = int(sum(upper_line) / len(upper_line))
    max_len = max(lines_len)
    return [lower_average, average, upper_average, max_len]


def construct_bucket_mean_gd(input_features, input_label, word_dict, label_dict):
    """
    Construct bucket by mean for greedy decode, word-level only
    """
    # encode and padding
    features = encode_safe(input_features, word_dict, word_dict['<unk>'])
    labels = encode(input_label, label_dict)
    labels = list(map(lambda t: [label_dict['<start>']] + list(t), labels))

    thresholds = calc_threshold_mean(features)

    return construct_bucket_gd(features, labels, thresholds, word_dict['<eof>'], label_dict['<pad>'])


def construct_bucket_mean_vb(input_features, input_label, word_dict, label_dict, caseless):
    """
    Construct bucket by mean for viterbi decode, word-level only
    """
    # encode and padding
    if caseless:
        input_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), input_features))

    features = encode_safe(input_features, word_dict, word_dict['<unk>'])
    labels = encode(input_label, label_dict)
    labels = list(map(lambda t: [label_dict['<start>']] + list(t), labels))

    thresholds = calc_threshold_mean(features)

    return construct_bucket_vb(features, labels, thresholds, word_dict['<eof>'], label_dict['<pad>'], len(label_dict))

def construct_bucket_mean_vb_wc(word_features, input_label, label_dict, char_dict, word_dict, caseless):
    """
    Construct bucket by mean for viterbi decode, word-level and char-level
    """
    # encode and padding
    char_features = encode2char_safe(word_features, char_dict)
    fea_len = [list(map(lambda t: len(t) + 1, f)) for f in char_features]
    forw_features = concatChar(char_features, char_dict)

    labels = encode(input_label, label_dict)
    labels = list(map(lambda t: [label_dict['<start>']] + list(t), labels))

    thresholds = calc_threshold_mean(fea_len)

    if caseless:
        word_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), word_features))
    word_features = encode_safe(word_features, word_dict, word_dict['<unk>'])

    return construct_bucket_vb_wc(word_features, forw_features, fea_len, labels, thresholds, word_dict['<eof>'], char_dict['\n'], label_dict['<pad>'], len(label_dict))

def construct_bucket_vb_wc(word_features, forw_features, fea_len, input_labels, thresholds, pad_word_feature, pad_char_feature, pad_label, label_size):
    """
    Construct bucket by thresholds for viterbi decode, word-level and char-level
    """
    # construct corpus for language model pre-training
    forw_corpus = [pad_char_feature] + list(reduce(lambda x, y: x + [pad_char_feature] + y, forw_features)) + [pad_char_feature]
    back_corpus = forw_corpus[::-1]
    # two way construct, first build the bucket, then calculate padding length, then do the padding
    buckets = [[[], [], [], [], [], [], [], []] for ind in range(len(thresholds))]
    # forw, forw_ind, back, back_in, label, mask
    buckets_len = [0 for ind in range(len(thresholds))]

    # thresholds is the padded length for fea
    # buckets_len is the padded length for char
    for f_f, f_l in zip(forw_features, fea_len):
        cur_len_1 = len(f_l) + 1
        idx = 0
        while thresholds[idx] < cur_len_1:
            idx += 1
        tmp_concat_len = len(f_f) + thresholds[idx] - len(f_l)
        if buckets_len[idx] < tmp_concat_len:
            buckets_len[idx] = tmp_concat_len

    # calc padding
    for f_f, f_l, w_f, i_l in zip(forw_features, fea_len, word_features, input_labels):
        cur_len = len(f_l)
        idx = 0
        cur_len_1 = cur_len + 1
        while thresholds[idx] < cur_len_1:
            idx += 1

        padded_feature = f_f + [pad_char_feature] * (buckets_len[idx] - len(f_f))  # pad feature with <'\n'>, at least one

        padded_feature_len = f_l + [1] * (thresholds[idx] - len(f_l)) # pad feature length with <'\n'>, at least one
        padded_feature_len_cum = list(itertools.accumulate(padded_feature_len)) # start from 0, but the first is ' ', so the position need not to be -1
        buckets[idx][0].append(padded_feature) # char
        buckets[idx][1].append(padded_feature_len_cum)
        buckets[idx][2].append(padded_feature[::-1])
        buckets[idx][3].append([buckets_len[idx] - 1] + [buckets_len[idx] - 1 - tup for tup in padded_feature_len_cum[:-1]])
        buckets[idx][4].append(w_f + [pad_word_feature] * (thresholds[idx] - cur_len)) #word
        buckets[idx][5].append([i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, cur_len)] + [i_l[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (thresholds[idx] - cur_len_1))  # has additional start, label
        buckets[idx][6].append([1] * cur_len_1 + [0] * (thresholds[idx] - cur_len_1))  # has additional start, mask
        buckets[idx][7].append([len(f_f) + thresholds[idx] - len(f_l), cur_len_1])
    bucket_dataset = [CRFDataset_WC(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]),
                                    torch.LongTensor(bucket[2]), torch.LongTensor(bucket[3]),
                                    torch.LongTensor(bucket[4]), torch.LongTensor(bucket[5]),
                                    torch.ByteTensor(bucket[6]), torch.LongTensor(bucket[7])) for bucket in buckets]
    return bucket_dataset, forw_corpus, back_corpus


def construct_bucket_vb(input_features, input_labels, thresholds, pad_feature, pad_label, label_size):
    """
    Construct bucket by thresholds for viterbi decode, word-level only
    """
    buckets = [[[], [], []] for _ in range(len(thresholds))]
    for feature, label in zip(input_features, input_labels):
        cur_len = len(feature)
        idx = 0
        cur_len_1 = cur_len + 1
        while thresholds[idx] < cur_len_1:
            idx += 1
        buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))
        buckets[idx][1].append([label[ind] * label_size + label[ind + 1] for ind in range(0, cur_len)] + [
            label[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (
                                       thresholds[idx] - cur_len_1))
        buckets[idx][2].append([1] * cur_len_1 + [0] * (thresholds[idx] - cur_len_1))
    bucket_dataset = [CRFDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.ByteTensor(bucket[2]))
                      for bucket in buckets]
    return bucket_dataset


def construct_bucket_gd(input_features, input_labels, thresholds, pad_feature, pad_label):
    """
    Construct bucket by thresholds for greedy decode, word-level only
    """
    buckets = [[[], [], []] for ind in range(len(thresholds))]
    for feature, label in zip(input_features, input_labels):
        cur_len = len(feature)
        cur_len_1 = cur_len + 1
        idx = 0
        while thresholds[idx] < cur_len_1:
            idx += 1
        buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))
        buckets[idx][1].append(label[1:] + [pad_label] * (thresholds[idx] - cur_len))
        buckets[idx][2].append(label + [pad_label] * (thresholds[idx] - cur_len_1))
    bucket_dataset = [CRFDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.LongTensor(bucket[2])) for bucket in buckets]
    return bucket_dataset


def find_length_from_feats(feats, feat_to_ix):
    """
    find length of unpadded features based on feature
    """
    end_position = len(feats) - 1
    for position, feat in enumerate(feats):
        if feat.data[0] == feat_to_ix['<eof>']:
            end_position = position
            break
    return end_position + 1


def find_length_from_labels(labels, label_to_ix):
    """
    find length of unpadded features based on labels
    """
    end_position = len(labels) - 1
    for position, label in enumerate(labels):
        if label == label_to_ix['<pad>']:
            end_position = position
            break
    return end_position


def revlut(lut):
    return {v: k for k, v in lut.items()}


# Turn a sequence of IOB chunks into single tokens
def iob_to_spans(sequence, lut, strict_iob2=False):
    """
    convert to iob to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (
                            label, current[0], i))

                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)

# Turn a sequence of IOBES chunks into single tokens
def iobes_to_spans(sequence, lut, strict_iob2=False):
    """
    convert to iobes to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):

            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):

            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')

        elif label.startswith('E-'):

            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None

            else:
                current = [label.replace('E-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def fill_y(nc, yidx):
    """
    fill y to dense matrix
    """
    batchsz = yidx.shape[0]
    siglen = yidx.shape[1]
    dense = np.zeros((batchsz, siglen, nc), dtype=np.int)
    for i in range(batchsz):
        for j in range(siglen):
            idx = int(yidx[i, j])
            if idx > 0:
                dense[i, j, idx] = 1

    return dense

def save_checkpoint(state, track_list, filename):
    """
    save checkpoint
    """
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
