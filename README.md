# x

First download CIFAR-10 python version from https://www.cs.toronto.edu/~kriz/cifar.html to obtain folder cifar-10-batches-py/

to run:
`python3 train.py --save True --incept-start-epoch 10`

Make sure to change the names of saved models (or back them up somewhere) before running new models that will be saved, because the new saved models will overwrite the old saved models if old saved models have default names.

##Dependencies
pytorch v0.2
