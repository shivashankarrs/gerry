#Evaluating Debiasing Techniques for Intersectional Biases

## How to run codes

## Datasets
Please download the data used for this project from the [link](https://drive.google.com/file/d/1cZcedMWSctHV5wYZ6qO8Fo7XjOc7sdhY/view?usp=sharing), which includes the following files:

* hatespeech data
** datasets/hatespeech/test.tsv
** datasets/hatespeech/train.tsv
** datasets/hatespeech/valid.tsv

* biasbios raw data + labels + private attributes
** datasets/bios/emnlp_train_bios_twoclass.pickle 
** datasets/bios/emnlp_dev_bios_twoclass.pickle
** datasets/bios/emnlp_test_bios_twoclass.pickle
* biasbios embeddings used to train debiased models
** datasets/bios/emnlp_train_cls_tc.npy
** datasets/bios/emnlp_dev_cls_tc.npy
** datasets/bios/emnlp_test_cls_tc.npy

## Citation

```@inproceedings{subramanian2021evaluating,
  title={Evaluating Debiasing Techniques for Intersectional Biases},
  author={Subramanian, Shivashankar and Han, Xudong and Baldwin, Timothy and Cohn, Trevor and Frermann, Lea},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={2492--2498},
  year={2021}
}```