<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## WWW'21 HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering

Authors: Jianing Sun*, Zhaoyue Cheng*, Saba Zuberi, Felipe Perez, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)  
[[paper](http://www.cs.toronto.edu/~mvolkovs/www2021_hgcf.pdf)]


<a name="Environment"/>

## Environment:

The code was developed and tested on the following python environment:
```
python 3.7.7
pytorch 1.5.1
scikit-learn 0.23.2
numpy 1.19.1
scipy 1.5.4
tqdm 4.48.2
```
<a name="instructions"/>

## Instructions:

1. Preprocess the [Amazon datasets](http://jmcauley.ucsd.edu/data/amazon) (CDs and Vinyl, Books), [Yelp2020 dataset](https://www.yelp.com/dataset/download), run this command:
```
python utils.preprocessing.py --dataset [Amazon_CD|Amazon_Book|yelp] --read_path your_raw_data_file 
```

2. Train and evaluation HGCF:
- `config.py` for training on `Amazon-CD` dataset:
```
config_args = {
    'training_config': {
        'log': (None, 'None for no logging'),
        'lr': (0.001, 'learning rate'),
        'batch-size': (10000, 'batch size'),
        'epochs': (500, 'maximum number of epochs to train for'),
        'weight-decay': (0.005, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
    },
    'model_config': {
        'embedding_dim': (50, 'user item embedding dimension'),
        'scale': (0.1, 'scale for init'),
        'dim': (50, 'embedding dimension'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
        'num-layers': (4,  'number of hidden layers in encoder'),
        'margin': (0.1, 'margin value in the metric learning loss'),
    },
    'data_config': {
        'dataset': ('Amazon-CD', 'which dataset to use'),
        'num_neg': (1, 'number of negative samples'),
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}
```

- `config.py` for training on `Amazon-Book` dataset:
```
config_args = {
    'training_config': {
        'log': (None, 'None for no logging'),
        'lr': (0.001, 'learning rate'),
        'batch-size': (10000, 'batch size'),
        'epochs': (500, 'maximum number of epochs to train for'),
        'weight-decay': (0.0005, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
    },
    'model_config': {
        'embedding_dim': (50, 'user item embedding dimension'),
        'scale': (0.1, 'scale for init'),
        'dim': (50, 'embedding dimension'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
        'num-layers': (4,  'number of hidden layers in encoder'),
        'margin': (0.1, 'margin value in the metric learning loss'),
    },
    'data_config': {
        'dataset': ('Amazon-Book', 'which dataset to use'),
        'num_neg': (1, 'number of negative samples'),
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}
```

- `config.py` for training on `yelp` dataset:
```
config_args = {
    'training_config': {
        'log': (None, 'None for no logging'),
        'lr': (0.001, 'learning rate'),
        'batch-size': (10000, 'batch size'),
        'epochs': (500, 'maximum number of epochs to train for'),
        'weight-decay': (0.001, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
    },
    'model_config': {
        'embedding_dim': (50, 'user item embedding dimension'),
        'scale': (0.1, 'scale for init'),
        'dim': (50, 'embedding dimension'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
        'num-layers': (4,  'number of hidden layers in encoder'),
        'margin': (0.2, 'margin value in the metric learning loss'),
    },
    'data_config': {
        'dataset': ('yelp', 'which dataset to use'),
        'num_neg': (1, 'number of negative samples'),
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}
```
<a name="citation"/>

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{sun2021hgcf,
      title={HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering},
      author={Jianing Sun, Zhaoyue Cheng, Saba Zuberi, Felipe Perez, Maksims Volkovs},
      booktitle={Proceedings of the International World Wide Web Conference},
      year={2021}
    }


