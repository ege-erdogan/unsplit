# UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label Inference Attacks Against Split Learning

**Abstract**: Training deep neural networks requires large scale data, which often forces users to work in a distributed or outsourced setting, accompanied with privacy concerns. Split learning framework aims to address this concern by splitting up the model among the client and the server. The idea is that since the server does not have access to client's part of the model, the scheme supposedly provides privacy. We show that this is not true via two novel attacks. (1) We show that an honest-but-curious split learning server, equipped only with the knowledge of the client neural network architecture, can recover the input samples and also obtain a functionally similar model to the client model, without the client being able to detect the attack. (2) Furthermore, we show that if split learning is used naively to protect the training labels, the honest-but-curious server can infer the labels with perfect accuracy. We test our attacks using three benchmark datasets and investigate various properties of the overall system that affect the attacks' effectiveness. Our results show that plaintext split learning paradigm can pose serious security risks and provide no more than a false sense of security.

https://arxiv.org/abs/2108.09033

## Code

We provide two Jupyter notebooks that interactively demonstrate the attacks: `model_inversion_stealing.ipynb` and `label_inference.ipynb`.

Furthermore, the repository includes two Python files that can be directly run to experiment with the attacks: `label_inference_demo.py` and `model_inversion_demo.py`. To ensure that you have the necessary dependencies, first run this command:
```
pip install -r requirements.txt
```

Once the requirements are satisfied, the demos can be run with the following commands:
```bash
python model_inversion_demo.py <dataset> <split_depth>
```
```bash
python label_inference_demo.py <dataset>
```
For now, our implementation supports MNIST(`mnist`), Fashion-MNIST(`f_mnist`), and CIFAR10(`cifar`) datasets for benchmarking. The `split_depth` field corresponds to the depth of the client model. Enter a value between `1` and `6` for MNIST datasets and a value between `1` and `8` for the CIFAR10 dataset.

The `unsplit` directory contains the following files that implement the attacks:
* `attacks.py`: Main implementation of the two attacks. 
* `models.py`: Models for testing purposes.
* `util.py`: Various helper methods. 


## Cite Our Work
```
@inproceedings{10.1145/3559613.3563201,
author = {Erdo\u{g}an, Ege and K\"{u}p\c{c}\"{u}, Alptekin and \c{C}i\c{c}ek, A. Erc\"{u}ment},
title = {UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label Inference Attacks against Split Learning},
year = {2022},
isbn = {9781450398732},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3559613.3563201},
doi = {10.1145/3559613.3563201},
booktitle = {Proceedings of the 21st Workshop on Privacy in the Electronic Society},
pages = {115–124},
numpages = {10},
keywords = {machine learning, split learning, label leakage, model inversion, model stealing, data privacy},
location = {Los Angeles, CA, USA},
series = {WPES'22}
}
```
