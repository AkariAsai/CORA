# CORA: Cross-lingual Open-Retrieval Answer Generation
This is the official implementation of the following paper:     
Akari Asai, Xinyan Yu, Jungo Kasai and Hannaneh Hajishirzi. [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](). *Preptint*. 2021. 

In this paper, we introduce **CORA**, a unified multilingual open QA model that retrieves multilingual passages and use them to generate answers in target language. CORA consists of two components, [mDPR](mdpr/README.md) and [mGEN](mGEN/README.md). **mDPR** retrieves documents from multilingual document collections and **mGEN** generates the answer in the target languages directly instead of using any external machine translation or language-specific retrieval module. Our experimental results show state-of-the-arr results across two multilingual open QA dataset: [XOR QA](https://nlp.cs.washington.edu/xorqa/) and [MKQA](https://github.com/apple/ml-mkqa). 

To run **CORA**, you first need to preprocess Wikipedia using the codes in [wikipedia_preprocess](wikipedia_preprocess). Then you train [mDPR](mDPR) and [mGEN](mGEN).

![cora_image](fig/cora_overview.png)



**Code and pretrained models will be added soon. Please stay tuned!**

## Citations and Contact

If you find this codebase is useful or use in your work, please cite our paper.
```
@article{
asai2021cora,
title={One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval},
author={Akari Asai and Xinyan Yu and Jungo Kasai and Hannaneh Hajishirzi},
year={2021}
}
```
Please contact Akari Asai ([@AkariAsai](https://twitter.com/AkariAsai) on Twitter, akari[at]cs.washington.edu) for questions and suggestions.
