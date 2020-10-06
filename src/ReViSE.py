import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torchvision
import os

class VisAE(nn.Module):
    def __init__(self, input_size=2048, hidden_size=500, output_size=100, final_feature_size=50):
        super(VisAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

        self.transformation = nn.Sequential(
            nn.Linear(output_size, final_feature_size),
            nn.Tanh()
        )
        
    def forward(self, x):

        h = self.encoder(x)
        out = self.decoder(h)
        t_out = self.transformation(h)

        return h, out, t_out

class AttAE(nn.Module):
    def __init__(self, input_size=300, output_size=100, final_feature_size=50):
        super(AttAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_size, input_size),
            nn.ReLU(),
        )

        self.transformation = nn.Sequential(
            nn.Linear(output_size, final_feature_size),
            nn.Tanh()
        )
        
    def forward(self, x):

        h = self.encoder(x)
        out = self.decoder(h)
        t_out = self.transformation(h)

        return h, out, t_out


def contractive_loss(visout, emb, vis_hidden, criterion2, device, gamma):
    #reconstruction loss
    loss1 = criterion2(visout, emb)
    # calculate gradient of the hidden layer wrt to the emb, to get Jacobian of encoder
    vis_hidden.backward(torch.ones(vis_hidden.size()).to(device), retain_graph=True)
    loss2 = torch.sqrt(torch.sum(torch.pow(emb.grad,2)))
    emb.grad.data.zero_()
    loss = loss1 + (gamma*loss2) 
    return loss 


def MMDLoss(vis, att, k = 32):
    n_samples = int(vis.size()[0]) + int(att.size()[0])
    total = torch.cat([vis, att], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2_distance = ((total0-total1)**2).sum(2)
    kernel = torch.exp(-k*l2_distance)

    loss = kernel[:n_samples//2, : n_samples//2] + kernel[n_samples//2:, n_samples//2:] - kernel[:n_samples//2, n_samples//2:] - kernel[n_samples//2:, :n_samples//2]

    return torch.mean(loss)

def multi_class_hinge_loss(vis_trans_out, att_trans_out, target, cemb, margin=0.3):
    temp_vis = vis_trans_out.unsqueeze(1).expand(vis_trans_out.shape[0], cemb.shape[0], vis_trans_out.shape[1])
    temp_cemb = cemb.unsqueeze(0).expand(vis_trans_out.shape[0], cemb.shape[0], vis_trans_out.shape[1])
    scores = torch.sum(temp_vis*temp_cemb, axis=2)
    gt_scores = torch.sum(vis_trans_out*att_trans_out, axis=1)
    gt_scores = gt_scores.unsqueeze(1).expand(gt_scores.shape[0], cemb.shape[0])
    out = (scores - gt_scores + margin)
    out[out<0] = 0
    loss = (1/vis_trans_out.shape[0])*torch.sum(out)

    return loss