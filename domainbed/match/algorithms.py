# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.match import networks


ALGORITHMS = [
    'VAE',
    'CVAE',
    'ViTVQ',
]

CONDITIANAL_ALGS = [
    'CVAE'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a generation algorithm.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        self.encoder = None
        self.decoder = None
        self.z_prior = None
        self.optimizer = None

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # sample from standard normal distribution

        return mu + eps * std

    def forward(self, x, y=None, d=None):
        raise NotImplementedError

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def get_match_var(self, x):
        raise NotImplementedError

    def reconstruct(self, x, y=None):
        raise NotImplementedError

    def inference(self, z, y=None):
        raise NotImplementedError
    
    def get_z(self, x):
        raise NotImplementedError


class VAE(Algorithm):
    """
    VAE that generate x from z 
    and infer z from x
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VAE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        try:
            self.encoder = networks.Encoder(input_shape, num_classes, 
                            self.hparams, conditional=False, use_mlp=True)
            self.decoder = networks.Decoder(input_shape, num_classes, 
                            self.hparams, conditional=False)
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        except:
            self.encoder = None
            self.decoder = None
            self.optimizer = None

    def forward(self, x, y=None, d=None):
        
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def update(self, minibatches, unlabeled=None, use_cf=False):

        all_x = torch.cat([x for x,y in minibatches])
        batch_size = all_x.size(0)

        recon_x, means, log_var, z = self.forward(all_x)

        # use Gaussian noise N(0, 1) and prior N(0, 1)
        mse = 0.5 * F.mse_loss(recon_x.view(batch_size, -1), 
                all_x.view(batch_size, -1), reduction='sum') / batch_size
        kld = -0.5 * torch.sum(1 + log_var - log_var.exp() - means.pow(2)) / batch_size
        loss = mse + kld

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'mse': mse.item(), 'kld': kld.item()}

    def reconstruct(self, x, y=None):

        means, log_var = self.encoder(x)
        recon_x = self.decoder(means)

        return recon_x

    def inference(self, z, y=None):

        recon_x = self.decoder(z)

        return {'recon_x': recon_x}
    
    def get_match_var(self, x):
        means, log_var = self.encoder(x)
        return means


class CVAE(Algorithm):
    """
    conditional VAE that generate x from y and z_d 
    with z_d spuriously associated with y
    and infer z_d from x
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CVAE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.num_domains = num_domains
        try:
            self.encoder = networks.Encoder(input_shape, num_classes, num_domains, 
                            self.hparams, conditional=True, use_mlp=True)
            self.decoder = networks.Decoder(input_shape, num_classes, 
                            self.hparams, conditional=True)
            self.z_prior = networks.Cond_Prior(num_classes, num_domains, 
                            self.hparams, domain_vary=True, 
                            distribution = hparams['distribution'])
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        except:
            self.encoder = None
            self.decoder = None
            self.z_prior = None
            self.optimizer = None

    def forward(self, x, y, d):
        
        prior_means, prior_log_var = self.z_prior(y, d)
        means, log_var = self.encoder(x, y, d)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, y)

        return recon_x, means, log_var, z, prior_means, prior_log_var

    def update(self, minibatches, unlabeled=None, use_cf=False):
        mse = 0.0
        kld = 0.0
        matched_cf_mse = 0.0
        num_envs = len(minibatches)

        for d, data in enumerate(minibatches):
            if use_cf:
                x,y,cf_x,cf_y = data
            else:
                x,y = data
            batch_size = x.size(0)
            recon_x, means, log_var, z, prior_means, prior_log_var = self.forward(x, y, d)
            
            # use Gaussian noise N(0, 1)
            mse += 0.5 * F.mse_loss(recon_x.view(batch_size, -1), 
                    x.view(batch_size, -1), reduction='sum') / (num_envs*batch_size)
            kld += -0.5 * torch.sum(1 + log_var - prior_log_var - 
                    (log_var.exp() + (means - prior_means).pow(2))
                    /prior_log_var.exp()) / (num_envs*batch_size)
            if use_cf:
                matched_mask = cf_y != -1
                if torch.all(~matched_mask):
                    continue
                matched_cf_x = cf_x[matched_mask]
                matched_cf_y = cf_y[matched_mask]
                matched_z = z[matched_mask]
                matched_batch_size = len(matched_cf_x)
                # reconstruct counterfactual example use the same latent 
                # as the factual exmaple
                matched_cf_result = self.inference(matched_z, matched_cf_y)
                matched_cf_mse += 0.5 * F.mse_loss(
                    matched_cf_result['recon_x'].view(matched_batch_size, -1), 
                    matched_cf_x.view(matched_batch_size, -1), reduction='sum'
                    ) / (num_envs*matched_batch_size)
        
        loss = mse + matched_cf_mse + self.hparams['kl_weight'] * kld

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if matched_cf_mse > 0:
            return {'loss': loss.item(), 'mse': mse.item(), 'kld': kld.item(), 
                    'cf_mse': matched_cf_mse.item()}
        else:
            return {'loss': loss.item(), 'mse': mse.item(), 'kld': kld.item(), 
                    'cf_mse': matched_cf_mse}

    def get_z(self, x, y, d):
        means, log_var = self.encoder(x, y, d)
        return means

    def inference(self, z, y):

        recon_x = self.decoder(z, y)

        return {'recon_x': recon_x}

    def reconstruct(self, x, y, d):

        means, log_var = self.encoder(x, y, d)
        recon_x = self.decoder(means, y)

        return recon_x
    
    def get_match_var(self, x, y, d):
        return self.get_z(x, y, d)

    def normal_log_pdf(self, mean, log_var, z):
        log_prob = -0.5*log_var - 0.5*((z-mean)/torch.sqrt(torch.exp(log_var)))**2
        return torch.sum(log_prob, dim=-1)

    def propensity_score(self, x, d, p_y=None): 
        log_p_z_y = []
        for i in range(self.num_classes):
            y = torch.tensor([i]*len(x), dtype=torch.int64, device=x.device)
            z = self.get_z(x, y, d)
            prior_means, prior_log_var = self.z_prior(y, d)
            log_p_z_y.append(self.normal_log_pdf(prior_means, prior_log_var, z))
        log_p_z_y = torch.stack(log_p_z_y, dim=1)

        p_y_z = []
        for i in range(self.num_classes):
            diff = torch.exp(log_p_z_y - log_p_z_y[:,i].unsqueeze(1)) # to avoid overflow
            if p_y is not None:
                diff = diff * (p_y/p_y[i]).unsqueeze(0)
            p_y_z.append(1/torch.sum(diff, dim=1))
        p_y_z = torch.stack(p_y_z, dim=1)
        p_y_z[p_y_z < 1e-30] = 1e-30 # avoid zeros
        return p_y_z