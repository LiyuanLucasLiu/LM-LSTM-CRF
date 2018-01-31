
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
    parser.add_argument('--test_file', default='', help='path to test file, if set to none, would use test_file path in the checkpoint file')
    args = parser.parse_args()

    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)


    # load corpus

    if args.test_file:
        with codecs.open(args.test_file, 'r', 'utf-8') as f:
            test_lines = f.readlines()
    else:
        with codecs.open(jd['test_file'], 'r', 'utf-8') as f:
            test_lines = f.readlines()

    # converting format

    test_features, test_labels = utils.read_corpus(test_lines)

    # construct dataset
    test_dataset = utils.construct_bucket_mean_vb(test_features, test_labels, f_map, l_map, jd['caseless'])
    
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

        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(ner_model, test_dataset_loader)

        print(jd['checkpoint'] + ' test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (test_f1, test_rec, test_pre, test_acc))

    else:

        test_acc = evaluator.calc_score(ner_model, test_dataset_loader)

        print(jd['checkpoint'] + ' test_acc: %.4f\n' % (test_acc))
