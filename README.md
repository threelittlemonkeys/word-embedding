# Word Embeddings

PyTorch implementations of word embedding models.

- The Continuous Bag-of-Words model (CBOW)
- The Skip-Gram model

## Usage

Training data should be formatted as below:
```
word word ...
word word ...
...
```

To prepare data:
```
python prepare.py training_data
```

To train:
```
python train.py model vocab training_data.csv num_epoch
```

To evaluate:
```
python evaluate.py model.epochN vocab word
```
