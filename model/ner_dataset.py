"""
.. module:: datasets
    :synopsis: datasets
 
.. moduleauthor:: Liyuan Liu
"""

from torch.utils.data import Dataset


class CRFDataset(Dataset):
    """Dataset Class for word-level model 

    args: 
        data_tensor (ins_num, seq_length): words 
        label_tensor (ins_num, seq_length): labels
        mask_tensor (ins_num, seq_length): padding masks
    """
    def __init__(self, data_tensor, label_tensor, mask_tensor):
        assert data_tensor.size(0) == label_tensor.size(0)
        assert data_tensor.size(0) == mask_tensor.size(0)
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.mask_tensor = mask_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index], self.mask_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

class CRFDataset_WC(Dataset):
    """Dataset Class for char-aware model 

    args: 
        forw_tensor (ins_num, seq_length): forward chars
        forw_index (ins_num, seq_length): index of forward chars
        back_tensor (ins_num, seq_length): backward chars
        back_index (ins_num, seq_length): index of backward chars
        word_tensor (ins_num, seq_length): words
        label_tensor (ins_num, seq_length): labels:
        mask_tensor (ins_num, seq_length): padding masks
        len_tensor (ins_num, 2): length of chars (dim0) and words (dim1)
    """
    def __init__(self, forw_tensor, forw_index, back_tensor, back_index, word_tensor, label_tensor, mask_tensor, len_tensor):
        assert forw_tensor.size(0) == label_tensor.size(0)
        assert forw_tensor.size(0) == mask_tensor.size(0)
        assert forw_tensor.size(0) == forw_index.size(0)
        assert forw_tensor.size(0) == back_tensor.size(0)
        assert forw_tensor.size(0) == back_index.size(0)
        assert forw_tensor.size(0) == word_tensor.size(0)
        assert forw_tensor.size(0) == len_tensor.size(0)
        self.forw_tensor = forw_tensor
        self.forw_index = forw_index
        self.back_tensor = back_tensor
        self.back_index = back_index
        self.word_tensor = word_tensor
        self.label_tensor = label_tensor
        self.mask_tensor = mask_tensor
        self.len_tensor = len_tensor

    def __getitem__(self, index):
        return self.forw_tensor[index], self.forw_index[index], self.back_tensor[index], self.back_index[index], self.word_tensor[index], self.label_tensor[index], self.mask_tensor[index], self.len_tensor[index]

    def __len__(self):
        return self.forw_tensor.size(0)
