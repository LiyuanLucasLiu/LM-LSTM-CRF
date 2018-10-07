.. LM-LSTM-CRF documentation master file, created by
   sphinx-quickstart on Thu Sep 14 03:49:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/LiyuanLucasLiu/LM-LSTM-CRF

LM-LSTM-CRF documentation
=========================

**Check Our New NER Toolkit🚀🚀🚀**

- **Inference**:

  - `LightNER <https://github.com/LiyuanLucasLiu/LightNER>`_: inference w. models pre-trained / trained w. *any* following tools, *efficiently*. 

- **Training**:

  - `LD-Net <https://github.com/LiyuanLucasLiu/LD-Net>`_: train NER models w. efficient contextualized representations.
  - `VanillaNER <https://github.com/LiyuanLucasLiu/Vanilla_NER>`_: train vanilla NER models w. pre-trained embedding.

- **Distant Training**:

  - `AutoNER <https://shangjingbo1226.github.io/AutoNER/>`_: train NER models w.o. line-by-line annotations and get competitive performance.

--------------------------

This project provides high-performance character-aware sequence labeling tools, including [Training](#usage), [Evaluation](#evaluation) and [Prediction](#prediction). 

Details about LM-LSTM-CRF can be accessed `here <http://arxiv.org/abs/1709.04109>`_, and the implementation is based on the PyTorch library. 

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/*

.. toctree::
   :maxdepth: 4
   :caption: Package Reference

   model


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
