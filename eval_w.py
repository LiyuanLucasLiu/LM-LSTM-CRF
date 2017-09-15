
from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lstm_crf import *
import model.utils as utils
from model.evaluator import eval_w

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating BLSTM-CRF')
    parser.add_argument('--load_arg', default='./checkpoint/soa/check_wc_p_char_lstm_crf.json', help='arg json file path')
    parser.add_argument('--load_check_point', default='./checkpoint/soa/check_wc_p_char_lstm_crf.model', help='checkpoint path')
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    args = parser.parse_args()

    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)


    # load corpus
    with codecs.open(jd['dev_file'], 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(jd['test_file'], 'r', 'utf-8') as f:
        test_lines = f.readlines()

    # converting format

    dev_features, dev_labels = utils.read_corpus(dev_lines)
    test_features, test_labels = utils.read_corpus(test_lines)

    # construct dataset
    dev_dataset = utils.construct_bucket_mean_vb(dev_features, dev_labels, f_map, l_map, jd['caseless'])
    test_dataset = utils.construct_bucket_mean_vb(test_features, test_labels, f_map, l_map, jd['caseless'])
    
    dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]

    # build model
    ner_model = LSTM_CRF(len(f_map), len(l_map), jd['embedding_dim'], jd['hidden'], jd['layers'], jd['drop_out'], large_CRF=jd['small_crf'])

    ner_model.load_state_dict(checkpoint_file['state_dict'])

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
        packer = CRFRepack(len(l_map), True)
    else:
        if_cuda = False
        packer = CRFRepack(len(l_map), False)

    evaluator = eval_w(packer, l_map, args.eva_matrix)

    if 'f' in args.eva_matrix:

        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(ner_model, dev_dataset_loader)

        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(ner_model, test_dataset_loader)

        print(jd['checkpoint'] + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))

    else:

        dev_acc = evaluator.calc_score(ner_model, dev_dataset_loader)

        test_acc = evaluator.calc_score(ner_model, test_dataset_loader)

        print(jd['checkpoint'] + ' dev_acc: %.4f test_acc: %.4f\n' % (dev_acc, test_acc))
