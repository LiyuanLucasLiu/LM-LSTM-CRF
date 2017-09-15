# LM-LSTM-CRF [![Documentation Status](https://readthedocs.org/projects/lm-lstm-crf/badge/?version=latest)](http://lm-lstm-crf.readthedocs.io/en/latest/?badge=latest)

This project provides high-performance character-aware sequence labeling tools and tutorials. Model details can be accessed at [here](http://arxiv.org/abs/1709.04109), and the implementation is based on the PyTorch library.

LM-LSTM-CRF achieves F1 score of 91.71+/-0.10 on CoNLL03 NER task, without using any additional corpus.

The documents would be available soon.

## Quick Links

- [Model](#model-notes)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Benchmarks](#benchmarks)

## Model Notes

<p align="center"><img width="100%" src="docs/framework.png"/></p>

As visualized above, we use conditional random field (CRF) to capture labels' dependency, and adopt hierarchical LSTM to take char-level and word-level input. 
The char-level structure is further guided by language model, while pre-trained word embedding is leveraged in word-level.
Language model and sequence labeling make predictions at word-level, and are trained at the same time.
[Highway networks]("https://arxiv.org/abs/1507.06228") are used to transform output of char-level into different semantic spaces, which mediate these two tasks and allows language model to empower sequence labeling.

## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be very slow.

### PyTorch

The code based on PyTorch; you can find installation instructions [here](http://pytorch.org/). 

### Dependencies

The code is written in Python 3.6; its dependencies are in the file ```requirements.txt```. You can install these dependencies like this:
```
pip install -r requirements.txt
```

## Data

We mainly focus on CoNLL 2003 NER task, and the code takes its format as input. 
However, due to the license issue, we are restricted to distribute this dataset.
You should be able to get it [here](http://aclweb.org/anthology/W03-0419).
You can also search it on github, there might be someone who released it.

### Format

We assume the corpus is formatted as CoNLL03 NER corpus.
Specifically, empty lines are used as separators between sentences, and the separator between documents is a special line:
```
-DOCSTART- -X- -X- -X- O
```
Other lines contains words, labels and other fields. Word must be the first field, label mush be the last, and these fields are separated by space.
For example, WSJ portion of PTB POS tagging corpus should be corpus like:

```
-DOCSTART- -X- -X- -X- O

Pierre NNP
Vinken NNP
, ,
61 CD
years NNS
old JJ
, ,
will MD
join VB
the DT
board NN
as IN
a DT
nonexecutive JJ
director NN
Nov. NNP
29 CD
. .


```

## Usage

Here we provides implements for two models, one is LM-LSTM-CRF, the other is its variant, LSTM-CRF, which only contains the word-level structure and CRF.
```train_wc.py``` and ```eval_wc.py``` are scripts for LM-LSTM-CRF, while ```train_w.py``` and ```eval_w.py``` are scripts for LSTM-CRF.
The usage of these scripts can be accessed by ````-h````, e.g, 
```
python train_wc.py -h
```

The default running command for NER, Noun Phrase Chunking are:

- Named Entity Recognition (NER):
```
python train_wc.py --train_file ./data/ner/train.txt --dev_file ./data/ner/testa.txt --test_file ./data/ner/testb.txt --checkpoint ./checkpoint/ner_ --caseless --fine_tune --high_way --co_train
```

- Part-of-Speech (POS) Tagging:
```
python train_wc.py --train_file ./data/pos/train.txt --dev_file ./data/pos/testa.txt --test_file ./data/pos/testb.txt --eva_matrix a --checkpoint ./checkpoint/pos_ --lr 0.015 --caseless --fine_tune --high_way --co_train
```

- Noun Phrase (NP) Chunking:
```
python train_wc.py --train_file ./data/np/train.txt.iobes --dev_file ./data/np/testa.txt.iobes --test_file ./data/np/testb.txt.iobes --checkpoint ./checkpoint/np_ --caseless --fine_tune --high_way --co_train --least_iters 100
```

For other datasets or tasks, you may wanna try different stopping parameters, especially, for smaller dataset, you may want to set ```least_iters``` to a larger value; and for some tasks, if the speed of loss decreasing is too slow, you may want to set ```lr``` to a larger value.

## Benchmarks

Here we compare LM-LSTM-CRF with recent state-of-the art models on CoNLL00 Chunking, CoNLL03 NER, and WSJ PTB POS Tagging task.

### NER

When only trained the on the CoNLL03 English NER corpus, the results are summarized below:

|Model | Max(F1) | Mean(F1) | Std(F1) | Reported(F1) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| Lample et al. 2016 | 91.14 | 90.76 | 0.08 | 90.94 | 46 |
| Ma et al. 2016 | 91.67 | 91.37 | 0.17 | 91.21 | 7 |
| LM-LSTM-CRF | **91.85** | **91.71** | 0.10 | | 6 |

### POS

When only trained the on WSJ portion of PTB POS Tagging corpus, the results are summarized below:

|Model | Max(Acc) | Mean(Acc) | Std(Acc) | Reported(Acc) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| Lample et al. 2016 | 97.51 | 97.35 | 0.09 | | 37 |
| Ma et al. 2016 | 97.46 | 97.42 | 0.04 | 97.55 | 21 |
| LM-LSTM-CRF | **97.59** | **97.53** | 0.03 | | 16 |

### Chunking

When only trained on the CoNLL00 Chunking corpus, the results are summarized below:

|Model | Max(F1) | Mean(F1) | Std(F1) | Time(h) |
| ------------- |-------------| -----| -----| ----|
| Lample et al. 2016 | 94.49 | 94.37 | 0.07 | 26 |
| Ma et al. 2016 | 95.93 | 95.80 | 0.13 | 6|
| LM-LSTM-CRF | **96.13** | **95.96** | 0.08 | 5 |