import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torchvision
import os

class VisAE(nn.Module):
    def __init__(self, input_size, vis_hidden_size, hidden_size, output_size):
        super(VisAE, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_size, vis_hidden_size),
            nn.ReLU(),
            nn.Linear(vis_hidden_size, hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, vis_hidden_size),
            nn.ReLU(),
            nn.Linear(vis_hidden_size, input_size),
            nn.ReLU()
        )

        self.transformation = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        
    def forward(self, x):

        h = self.encoder(x)
        out = self.decoder(h)
        t_out = self.transformation(h)

        return h, out, t_out

class AE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AE, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

        self.verb_transformation = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):

        h = self.encoder(x)
        out = self.decoder(h)
        vt_out = self.verb_transformation(h)
        return h, out, vt_out
        
class PosMmen(nn.Module):
    def __init__(self, vis_input_size, text_input_size, vis_hidden_size, text_hidden_size, output_size):
        super(PosMmen, self).__init__()

        if vis_input_size == 2048:
            self.VisArch = VisAE(vis_input_size, vis_hidden_size, text_hidden_size, output_size)
        else:
            self.VisArch = AE(vis_input_size, text_hidden_size, output_size)
        
        if text_input_size == 1024:
            self.TextArch = VisAE(text_input_size, vis_hidden_size, text_hidden_size, output_size)
        else:
            self.TextArch = AE(text_input_size, text_hidden_size, output_size)
       
    def forward(self, vis_x, text_x):
        vh, vr, vo = self.VisArch(vis_x)
        th, tr, to = self.TextArch(text_x) 

        return vh, vr, vo, th, tr, to

class JPosMmen(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(JPosMmen, self).__init__()

        self.VisArch = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.Tanh(),
        )
        
        self.TextArch = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.Tanh(),
        )

    def forward(self, vis_x, text_x):
        vis_out = self.VisArch(vis_x)
        text_out = self.TextArch(text_x) 

        return vis_out, text_out


class Mmen(nn.Module):
    def __init__(self, vis_input_size, text_input_size, out_size):
        super(Mmen, self).__init__()
    
        self.VisArch = nn.Sequential(
            nn.Linear(vis_input_size, out_size), 
            nn.Tanh(),
        )
        
        if text_input_size == 1024:
            self.TextArch = nn.Sequential(
                nn.Linear(text_input_size, 512), 
                nn.Tanh(),
                nn.Linear(512, out_size), 
                nn.ReLU()
            )
        else:
            self.TextArch = nn.Sequential(
                nn.Linear(text_input_size, out_size), 
                nn.Tanh(),
            )

    def forward(self, vis_x, text_x):
        vis_out = self.VisArch(vis_x)
        text_out = self.TextArch(text_x) 

        return vis_out, text_out



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

def triplet_loss(vis_trans_out, target, device, margin = 0.1):
    
    #mining samples
    pos_samps = torch.zeros(vis_trans_out.shape).float().to(device)
    neg_samps = torch.zeros(vis_trans_out.shape).float().to(device)
    with torch.no_grad():
        for i in torch.unique(target):
            pos_cands_inds = np.argwhere(target == i).flatten()
            pos_cands = vis_trans_out[pos_cands_inds]
            temp_pos1 = pos_cands.unsqueeze(1).expand(pos_cands.shape[0], pos_cands.shape[0], pos_cands.shape[1])
            temp_pos2 = pos_cands.unsqueeze(0).expand(pos_cands.shape[0], pos_cands.shape[0], pos_cands.shape[1])
            pos_samps[pos_cands_inds] = pos_cands[torch.argmin(torch.sum(temp_pos1*temp_pos2, dim=2), dim=1)]
            
            neg_cands_inds = np.array(list(set(np.arange(target.shape[0])) - set(pos_cands_inds)))
            neg_cands = vis_trans_out[neg_cands_inds]
            temp_pos = pos_cands.unsqueeze(1).expand(pos_cands.shape[0], neg_cands.shape[0], pos_cands.shape[1])
            temp_neg = neg_cands.unsqueeze(0).expand(pos_cands.shape[0], neg_cands.shape[0], neg_cands.shape[1])
            neg_samps[pos_cands_inds] = neg_cands[torch.argmax(torch.sum(temp_pos*temp_neg, dim=2), dim=1)]
    
    temp_vis_trans_out = vis_trans_out.unsqueeze(1).expand(vis_trans_out.shape[0], pos_samps.shape[0], vis_trans_out.shape[1])
    temp_pos_samps = pos_samps.unsqueeze(0).expand(vis_trans_out.shape[0], pos_samps.shape[0], vis_trans_out.shape[1])
    temp_neg_samps = neg_samps.unsqueeze(0).expand(vis_trans_out.shape[0], pos_samps.shape[0], vis_trans_out.shape[1])
    pos_score = torch.sum((temp_vis_trans_out*temp_pos_samps), dim=2)
    neg_score = torch.sum((temp_vis_trans_out*temp_neg_samps), dim=2)
    fin_score = neg_score - pos_score + margin
    fin_score[fin_score<0] = 0
    loss = (1/vis_trans_out.shape[0])*torch.sum(fin_score)

    return loss
