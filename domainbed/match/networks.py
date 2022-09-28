# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

### CVAE utilities ###

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x.view(len(x), -1))
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class Encoder(nn.Module):

    def __init__(self, x_shape, num_classes, num_domains, 
                    hparams, conditional, use_mlp=True):

        super().__init__()
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.conditional = conditional
        self.hparams = hparams
        self.use_mlp = use_mlp

        if self.use_mlp:
            if self.conditional:
                input_shape = x_shape[0]*x_shape[1]*x_shape[2] + num_classes + num_domains
            else:
                input_shape = x_shape[0]*x_shape[1]*x_shape[2]
        else:
            input_shape = x_shape

        self.featurizer = MLP(input_shape, hparams["mlp_width"], hparams)

        if self.conditional and not self.use_mlp:
            self.class_embd = nn.Linear(num_classes, self.featurizer.n_outputs)
            self.domain_embd = nn.Linear(num_domains, self.featurizer.n_outputs)

        self.linear_means = nn.Linear(self.featurizer.n_outputs, self.hparams["latent_size"])
        self.linear_log_var = nn.Linear(self.featurizer.n_outputs, self.hparams["latent_size"])

    def forward(self, x, c=None, d=None):
        if self.conditional:
            d = torch.zeros_like(c) + d
            c = idx2onehot(c, n=self.num_classes)
            d = idx2onehot(d, n=self.num_domains)

        if self.use_mlp:
            if self.conditional:
                x = torch.cat((x.view(len(x), -1), c, d), dim=-1)
            else:
                x = x.view(len(x), -1)
        x = self.featurizer(x)

        if self.conditional and not self.use_mlp:
            c_embd = self.class_embd(c)
            d_embd = self.domain_embd(d)
            x += c_embd + d_embd

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, x_shape,  num_classes, hparams, conditional, num_latent=1):

        super().__init__()
        
        self.num_classes = num_classes
        self.conditional = conditional
        self.hparams = hparams

        if self.conditional:
            input_shape = self.hparams["latent_size"] * num_latent + num_classes
        else:
            input_shape = self.hparams["latent_size"] * num_latent

        self.featurizer = MLP(input_shape, hparams["mlp_width"], hparams)

        self.out_layer = nn.Linear(self.featurizer.n_outputs, x_shape[0]*x_shape[1]*x_shape[2])

    def forward(self, z, c=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_classes)
            z = torch.cat((z, c), dim=-1)

        x = self.featurizer(z)
        x = self.out_layer(x)

        return torch.sigmoid(x)

class Cond_Prior(nn.Module):

    def __init__(self, num_classes, num_domains, hparams, latent_size=None, domain_vary=True, distribution='Gaussian'):

        super().__init__()
        
        self.num_classes = num_classes
        self.hparams = hparams
        self.domain_vary = domain_vary
        self.distribution = distribution

        if self.distribution == 'Gaussian':
            if self.domain_vary:
                self.linear_means = nn.ModuleList()
                self.linear_log_var = nn.ModuleList()
                for i in range(num_domains):
                    self.linear_means.append(nn.Linear(num_classes, self.hparams["latent_size"]))
                    self.linear_log_var.append(nn.Linear(num_classes, self.hparams["latent_size"]))
            else:
                self.linear_means = nn.Linear(num_classes, self.hparams["latent_size"])
                self.linear_log_var = nn.Linear(num_classes, self.hparams["latent_size"])

        elif self.distribution == 'Gaussian_fixed_variance':
            if self.domain_vary:
                self.linear_means = nn.ModuleList()
                for i in range(num_domains):
                    self.linear_means.append(nn.Linear(num_classes, self.hparams["latent_size"]))
            else:
                self.linear_means = nn.Linear(num_classes, self.hparams["latent_size"])
            self.log_var = 0.0

        elif self.distribution == 'Multinomial':
            self.latent_size = latent_size
            self.out_size = latent_size[0] * latent_size[1] * self.hparams['num_embeddings']
            if self.domain_vary:
                self.logits = nn.ModuleList()
                for i in range(num_domains):
                    self.logits.append(nn.Linear(num_classes, self.out_size))
            else:
                self.logits = nn.Linear(num_classes, self.out_size)

        elif self.distribution == 'Uniform':
            self.log_prob = -torch.log(torch.tensor(self.hparams['num_embeddings']).float())
        
        else:
            print(self.distribution, " distribution is not implemented.")
            exit(1)

    def forward(self, c, d=None):

        c = idx2onehot(c, n=self.num_classes)

        if self.distribution == 'Gaussian':
            if self.domain_vary:
                means = self.linear_means[d](c)
                log_vars = self.linear_log_var[d](c)
            else:
                means = self.linear_means(c)
                log_vars = self.linear_log_var(c)
            return means, log_vars
        
        elif self.distribution == 'Gaussian_fixed_variance':
            if self.domain_vary:
                means = self.linear_means[d](c)
            else:
                means = self.linear_means(c)
            return means, torch.zeros_like(means, device = means.device)

        elif self.distribution == 'Multinomial':
            if self.domain_vary:
                log_probs = F.log_softmax(self.logits[d](c), -1).view(
                            -1, self.latent_size[0], self.latent_size[1], self.hparams['num_embeddings'])
            else:
                log_probs = F.log_softmax(self.logits(c), -1).view(
                            -1, self.latent_size[0], self.latent_size[1], self.hparams['num_embeddings'])
            return log_probs

        elif self.distribution == 'Uniform':
            return self.log_prob

        else:
            print(self.distribution, " distribution is not implemented.")
            exit(1)