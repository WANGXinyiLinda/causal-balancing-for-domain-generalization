# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST', 'ColoredMNIST_10class']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam('data_augmentation', False, lambda r: r.choice([True, False]))
    _hparam('simple_augmentation', False, lambda r: r.choice([True, False]))
    _hparam('class_balanced', False, lambda r: r.choice([True, False]))
    _hparam('reduce_dim', False, lambda r: r.choice([True, False]))
    _hparam('normalize', False, lambda r: r.choice([True, False]))
    _hparam('holdout_fraction', 0.0, lambda r: r.uniform(0, 1))
    _hparam('topk', 1, lambda r: r.choice([0.2, 0.3, 0.4]))
    _hparam('distance_func', 'kld', lambda r: r.choice(['l2', 'linf', 'kld']))
    _hparam('threshold', False, lambda r: r.choice([True, False]))
    _hparam('mlp_dropout', 0.0, lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('kl_weight', 1, lambda r: 10**r.uniform(-4.5, -2.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
        _hparam('image_size', 28, lambda r: 28)
        _hparam('mlp_width', 512, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 2, lambda r: int(r.choice([3, 4, 5])))
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('distribution', 'Gaussian_fixed_variance', lambda r: 'Gaussian_fixed_variance')
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('image_size', 224, lambda r: int(r.choice([32, 64, 128])))
        _hparam('mlp_width', 1024, lambda r: int(2 ** r.uniform(8, 12)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([5, 6, 7])))
        _hparam('lr', 1e-4, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('distribution', 'Gaussian', lambda r: 'Gaussian')
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 9)))

    if dataset == 'ColoredMNIST':
        _hparam('noise_rate', 0.25, lambda r: r.choice([0., 0.25, 0.5]))
        _hparam('environments', [0.1, 0.2, 0.9], lambda r: [0.1, 0.2, 0.9])
        _hparam('latent_size', 3, lambda r: 3)
    elif dataset == 'ColoredMNIST_10class':
        _hparam('latent_size', 16, lambda r: 16)
    elif dataset == 'RotatedMNIST':
        _hparam('environments', [0, 15, 30, 45, 60, 75], lambda r: [0, 15, 30, 45, 60, 75])
        _hparam('latent_size', 16, lambda r: 16)
    elif dataset == 'VLCS':
        _hparam('latent_size', 7, lambda r: 7)
    elif dataset == 'PACS':
        _hparam('latent_size', 10, lambda r: 10)
    elif dataset == 'TerraIncognita':
        _hparam('latent_size', 14, lambda r: 14)
    else:
        _hparam('latent_size', 64, lambda r: 64)

    return hparams

def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
