# UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label Inference Attacks Against Split Learning

TODO: Abstract

TODO: Link to paper

## Code

The repository includes two Python files demonstrating our two attacks: `label_inference_demo.py` and `model_inversion_demo.py`. To ensure that you have the necessary dependencies, first run this command:
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
TODO: Citation
```
