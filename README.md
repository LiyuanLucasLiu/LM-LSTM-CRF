# LM-LSTM-CRF
Empower Sequence Labeling with Task-Aware Language Model

## Training

```
python train_nwc.py --checkpoint ./checkpoint/ner_ --gpu 0 --caseless --fine_tune --high_way --co_train
```

- Named Entity Recognition (NER)
```
python train_nwc.py --patience 15 --checkpoint ./checkpoint/w_${num}_ --gpu ${gpuid} --epoch 200 --lr 0.01 --lr_decay 0.05 --momentum 0.9 --caseless --fine_tune --mini_count 5 --char_hidden 300 --word_hidden 300 --pc_type w --high_way --co_train 2>> l_ner/out_${num}.txt
```

- Part-of-Speech (POS) Tagging
```
python train_nwc.py --train_file ./data/pos/train.txt --dev_file ./data/pos/testa.txt --test_file ./data/pos/testb.txt --eva_matrix a --checkpoint ./checkpoint/pos_ --gpu 1 --lr 0.015 --caseless --fine_tune --high_way --co_train
```

- Noun Phrase (NP) Chunking
```
python train_nwc.py --train_file ./data/np/train.txt.iobes --dev_file ./data/np/testa.txt.iobes --test_file ./data/np/testb.txt.iobes --checkpoint ./checkpoint/np_ --gpu 2 --caseless --fine_tune --high_way --co_train
```
