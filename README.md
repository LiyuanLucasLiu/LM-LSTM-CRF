# LM-LSTM-CRF [![Documentation Status](https://readthedocs.org/projects/lm-lstm-crf/badge/?version=latest)](http://lm-lstm-crf.readthedocs.io/en/latest/?badge=latest)

This project provides high-performance character-aware sequence labeling tools and tutorials. Model details can be accessed [here](http://arxiv.org/abs/1709.04109), and the implementation is based on the PyTorch library.

LM-LSTM-CRF achieves F1 score of 91.71+/-0.10 on the CoNLL 2003 NER dataset, without using any additional corpus or resource.

The documents would be available [here](http://lm-lstm-crf.readthedocs.io/en/latest/).

## Quick Links

- [Model](#model-notes)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Benchmarks](#benchmarks)

## Model Notes

<p align="center"><img width="100%" src="docs/framework.png"/></p>

As visualized above, we use conditional random field (CRF) to capture label dependencies, and adopt a hierarchical LSTM to leverage both char-level and word-level inputs. 
The char-level structure is further guided by a language model, while pre-trained word embeddings are leveraged in word-level.
The language model and the sequence labeling model are trained at the same time, and both make predictions at word-level.
[Highway networks]("https://arxiv.org/abs/1507.06228") are used to transform the output of char-level LSTM into different semantic spaces, and thus mediating these two tasks and allowing language model to empower sequence labeling.

## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.

### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/). 

### Dependencies

The code is written in Python 3.6. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:
```
pip3 install -r requirements.txt
```

## Data

We mainly focus on the CoNLL 2003 NER dataset, and the code takes its original format as input. 
However, due to the license issue, we are restricted to distribute this dataset.
You should be able to get it [here](http://aclweb.org/anthology/W03-0419).
You may also want to search online (e.g., Github), someone might release it accidentally.

### Format

We assume the corpus is formatted as same as the CoNLL 2003 NER dataset.
More specifically, **empty lines** are used as separators between sentences, and the separator between documents is a special line as below.
```
-DOCSTART- -X- -X- -X- O
```
Other lines contains words, labels and other fields. **Word** must be the **first** field, **label** mush be the **last**, and these fields are **separated by space**.
For example, the first several lines in the WSJ portion of the PTB POS tagging corpus should be like the following snippet.

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

Here we provide implementations for two models, one is **LM-LSTM-CRF** and the other is its variant, **LSTM-CRF**, which only contains the word-level structure and CRF.
```train_wc.py``` and ```eval_wc.py``` are scripts for LM-LSTM-CRF, while ```train_w.py``` and ```eval_w.py``` are scripts for LSTM-CRF.
The usages of these scripts can be accessed by the parameter ````-h````, i.e., 
```
python train_wc.py -h
python train_w.py -h
python eval_wc.py -h
python eval_w.py -h
```

The default running commands for NER and POS tagging, and NP Chunking are:

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

For other datasets or tasks, you may wanna try different stopping parameters, especially, for smaller dataset, you may want to set ```least_iters``` to a larger value; and for some tasks, if the speed of loss decreasing is too slow, you may want to increase ```lr```.

## Benchmarks

Here we compare LM-LSTM-CRF with recent state-of-the-art models on the CoNLL 2000 Chunking dataset, the CoNLL 2003 NER dataset, and the WSJ portion of the PTB POS Tagging dataset. All experiments are conducted on a GTX 1080 GPU.

### NER

When models are only trained on the CoNLL 2003 English NER dataset, the results are summarized as below.

|Model | Max(F1) | Mean(F1) | Std(F1) | Reported(F1) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| Lample et al. 2016 | 91.14 | 90.76 | 0.08 | 90.94 | 46 |
| Ma et al. 2016 | 91.67 | 91.37 | 0.17 | 91.21 | 7 |
| LM-LSTM-CRF | **91.85** | **91.71** | 0.10 | | 6 |

### POS

When models are only trained on the WSJ portion of the PTB POS Tagging dataset, the results are summarized as below.

|Model | Max(Acc) | Mean(Acc) | Std(Acc) | Reported(Acc) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| Lample et al. 2016 | 97.51 | 97.35 | 0.09 | | 37 |
| Ma et al. 2016 | 97.46 | 97.42 | 0.04 | 97.55 | 21 |
| LM-LSTM-CRF | **97.59** | **97.53** | 0.03 | | 16 |

### Chunking

When models are only trained on the CoNLL 2000 Chunking dataset, the results are summarized as below.

|Model | Max(F1) | Mean(F1) | Std(F1) | Time(h) |
| ------------- |-------------| -----| -----| ----|
| Lample et al. 2016 | 94.49 | 94.37 | 0.07 | 26 |
| Ma et al. 2016 | 95.93 | 95.80 | 0.13 | 6|
| LM-LSTM-CRF | **96.13** | **95.96** | 0.08 | 5 |


## Reference

```
@ARTICLE{2017arXiv170904109L,
  title = "{Empower Sequence Labeling with Task-Aware Neural Language Model}", 
  author = {{Liu}, L. and {Shang}, J. and {Xu}, F. and {Ren}, X. and {Gui}, H. and {Peng}, J. and {Han}, J.}, 
  journal = {	arXiv:1709.04109}, 
  year = 2017, 
}
```
