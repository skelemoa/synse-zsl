import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torchvision
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Encoder, self).__init__()
    
        layers = []
        for i in range(len(layer_sizes)-2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            
        self.model = nn.Sequential(*layers)

        self.mu_transformation = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        
        self.logvar_transformation = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        
        self.apply(weights_init)

    def forward(self, x):

        h = self.model(x)
        mu = self.mu_transformation(h)
        logvar = self.logvar_transformation(h)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            
        self.model = nn.Sequential(*layers)
        
        self.apply(weights_init)

    def forward(self, x):

        out = self.model(x)        
        return out
        
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        self.apply(weights_init)
    
    def forward(self, x):
        return self.model(x)

def reparameterize(mu, logvar):
    sigma = torch.exp(0.5*logvar)
    # eps = torch.randn_like(sigma)
    eps = torch.FloatTensor(sigma.size()[0], 1).normal_(0, 1).expand(sigma.size()).cuda()
    return eps*sigma + mu

def KL_divergence(mu, logvar):
    return 0.5*(torch.sum( - (mu**2) + 1 + logvar - torch.exp(logvar)))/mu.shape[0]

def Wasserstein_distance(mu1, logvar1, mu2, logvar2):
    return torch.sum(torch.sqrt(torch.sum((mu1 - mu2)**2, dim=1) + torch.sum((torch.sqrt(torch.exp(logvar1)) - torch.sqrt(torch.exp(logvar2)))**2, dim=1)))

def triplet_loss(latent_mean, target, device, margin = 0.1):
    
    #mining samples
    pos_samps = torch.zeros(latent_mean.shape).float().to(device)
    neg_samps = torch.zeros(latent_mean.shape).float().to(device)
    with torch.no_grad():
        for i in torch.unique(target):
            pos_cands_inds = np.argwhere(target == i).flatten()
            pos_cands = latent_mean[pos_cands_inds]
            temp_pos1 = pos_cands.unsqueeze(1).expand(pos_cands.shape[0], pos_cands.shape[0], pos_cands.shape[1])
            temp_pos2 = pos_cands.unsqueeze(0).expand(pos_cands.shape[0], pos_cands.shape[0], pos_cands.shape[1])
            pos_samps[pos_cands_inds] = pos_cands[torch.argmin(torch.sum(temp_pos1*temp_pos2, dim=2), dim=1)]
            
            neg_cands_inds = np.array(list(set(np.arange(target.shape[0])) - set(pos_cands_inds)))
            neg_cands = latent_mean[neg_cands_inds]
            temp_pos = pos_cands.unsqueeze(1).expand(pos_cands.shape[0], neg_cands.shape[0], pos_cands.shape[1])
            temp_neg = neg_cands.unsqueeze(0).expand(pos_cands.shape[0], neg_cands.shape[0], neg_cands.shape[1])
            neg_samps[pos_cands_inds] = neg_cands[torch.argmax(torch.sum(temp_pos*temp_neg, dim=2), dim=1)]
    
    temp_vis_trans_out = latent_mean.unsqueeze(1).expand(latent_mean.shape[0], pos_samps.shape[0], latent_mean.shape[1])
    temp_pos_samps = pos_samps.unsqueeze(0).expand(latent_mean.shape[0], pos_samps.shape[0], latent_mean.shape[1])
    temp_neg_samps = neg_samps.unsqueeze(0).expand(latent_mean.shape[0], pos_samps.shape[0], latent_mean.shape[1])
    pos_score = torch.sum((temp_vis_trans_out*temp_pos_samps), dim=2)
    neg_score = torch.sum((temp_vis_trans_out*temp_neg_samps), dim=2)
    fin_score = neg_score - pos_score + margin
    fin_score[fin_score<0] = 0
    loss = torch.mean(torch.sum(fin_score, axis=1))

    return loss