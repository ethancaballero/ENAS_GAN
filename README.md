# ENAS_GAN
###### collaboration between: [Ethan Caballero](https://github.com/ethancaballero) & [Aran Komatsuzaki](https://github.com/AranKomat)

This is an attempt to use [ENAS](https://arxiv.org/abs/1802.03268) (Efficient Neural Architecture Search) to find optimal GAN architectures. Unfortunately, ENAS seems to be unable to find useful GAN architectures.

First download CIFAR-10 python version from https://www.cs.toronto.edu/~kriz/cifar.html to obtain folder cifar-10-batches-py/

to run:
`python3 train.py --save True --incept-start-epoch 10`

to run ppo:
`python3 train.py --save True --incept-start-epoch 10 --ppo True`

Make sure to change the names of saved models (or back them up somewhere) before running new models that will be saved, because the new saved models will overwrite the old saved models if old saved models have default names.

## Dependencies
pytorch v0.2 or v0.3
