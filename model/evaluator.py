import torch
import numpy as np
import itertools

import model.utils as utils
from torch.autograd import Variable

from model.crf import CRFDecode_vb

class eval_batch:

    def __init__(self, packer, l_map):
        self.packer = packer
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)

    def reset(self):
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, decoded_data, target_data):
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length]
            best_path = decoded[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path.numpy(), gold.numpy())
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def calc_acc_batch(self, decoded_data, target_data):
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):
        if self.guess_count == 0:
            return 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy        

    def eval_instance(self, best_path, gold):
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = utils.iobes_to_spans(gold, self.r_l_map)
        gold_count = len(gold_chunks)

        guess_chunks = utils.iobes_to_spans(best_path, self.r_l_map)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

class eval_w(eval_batch):

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)
        
        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        ner_model.eval()
        self.reset()
                
        for feature, tg, mask in itertools.chain.from_iterable(dataset_loader):
            fea_v, _, mask_v = self.packer.repack_vb(feature, tg, mask)
            scores, _ = ner_model(fea_v)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        return self.calc_s()

class eval_wc(eval_batch):

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)
        
        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        ner_model.eval()
        self.reset()

        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v in itertools.chain.from_iterable(dataset_loader):
            f_f, f_p, b_f, b_p, w_f, _, mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v)
            scores = ner_model(f_f, f_p, b_f, b_p, w_f)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        return self.calc_s()