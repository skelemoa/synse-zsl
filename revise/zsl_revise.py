import argparse
import time
import shutil
import os
import os.path as osp
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torchvision.models as models
from data_cnn60 import NTUDataLoaders, AverageMeter, make_dir, get_cases, get_num_classes
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import torch.nn.functional as F
from ReViSE import VisAE, AttAE, contractive_loss, MMDLoss, multi_class_hinge_loss

parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="No of unseen classes")
parser.add_argument('--st', type=str, help="Type of split")
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--wdir', type=str, help="directory to save weights path")
parser.add_argument('--le', type=str, help="language embedding model")
parser.add_argument('--ve', type=str, help="visual embedding model")
parser.add_argument('--phase', type=str, help="train or val")
parser.add_argument('--gpu', type=str, help="gpu device number")
parser.add_argument('--ntu', type=int, help="no of classes")
args = parser.parse_args()

gpu = args.gpu
ss = args.ss
st = args.st
dataset_path = args.dataset
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_classes = args.ntu
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda")
print(torch.cuda.device_count())

if not os.path.exists('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir):
    os.mkdir('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir)
if not os.path.exists('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le):
    os.mkdir('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le)

criterion2 = nn.MSELoss().to(device)

if le == 'bert':
    att_input_size = 1024
    att_intermediate_size = 512
    att_hidden_size = 100
    attae = VisAE(att_input_size, att_intermediate_size, att_hidden_size).to(device)
elif le == 'w2v':
    att_input_size = 300
    att_hidden_size = 100
    attae = AttAE(att_input_size, att_hidden_size).to(device)
else:
    pass
# attae.load_state_dict(torch.load(zattae_checkpoint)['revise_state_dict'], strict=False)
attae_optimizer = optim.Adam(attae.parameters(),lr=1e-4, weight_decay = 0.001)
attae_scheduler = ReduceLROnPlateau(attae_optimizer, mode='max', factor=0.1, patience=25, cooldown=3, verbose=True)

if ve == 'vacnn':
    vis_input_size = 2048
    vis_intermediate_size = 512
    vis_hidden_size = 100
    visae = VisAE(vis_input_size, vis_intermediate_size, vis_hidden_size).to(device)
elif ve == 'shift':
    vis_input_size = 256
    vis_hidden_size = 100
    visae = AttAE(vis_input_size, vis_hidden_size).to(device)
else:
    pass

# visae.load_state_dict(torch.load(zvisae_checkpoint)['revise_state_dict'], strict=False)
visae_optimizer = optim.Adam(visae.parameters(), lr=1e-4)
visae_scheduler = ReduceLROnPlateau(visae_optimizer, mode='max', factor=0.1, patience=25, cooldown=3, verbose=True)
print("Loaded Revise Model")

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_train_loader(1024, 8)
zsl_loader = ntu_loaders.get_val_loader(1024, 8)
val_loader = ntu_loaders.get_test_loader(1024, 8)
train_size = ntu_loaders.get_train_size()
zsl_size = ntu_loaders.get_test_size()
val_size = ntu_loaders.get_val_size()
print('Train on %d samples, validate on %d samples' % (train_size, zsl_size))


if phase == 'val':
    gzsl_inds = np.load('../resources/label_splits/'+st+'s'+str(num_classes - ss)+'.npy')
    unseen_inds = np.load('../resources/label_splits/'+st+'v'+str(ss)+'_0.npy')
    seen_inds = np.load('../resources/label_splits/'+st+'s'+str(num_classes - ss - ss)+'_0.npy')
else:
    gzsl_inds = np.arange(num_classes)
    unseen_inds = np.load('../resources/label_splits/'+st+'u'+str(ss)+'.npy')
    seen_inds = np.load('../resources/label_splits/'+st+'s'+str(num_classes - ss)+'.npy')

labels = np.load('../resources/labels.npy')
unseen_labels = labels[unseen_inds]
seen_labels = labels[seen_inds]


def get_emb(model, words):
    emb = np.zeros([300])
    for word in words.split():
        emb += model[word]
    emb /= len(words.split())
    
    return emb

s2v_labels = torch.from_numpy(np.load('../resources/ntu' + str(num_classes) + '_' + le + '_labels.npy')).view([num_classes, att_input_size])
s2v_labels = s2v_labels/torch.norm(s2v_labels, dim = 1).view([num_classes, 1]).repeat([1, att_input_size])

unseen_s2v_labels = s2v_labels[unseen_inds, :]
seen_s2v_labels = s2v_labels[seen_inds, :]


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def lstm_accuracy(nout, vout, ntarget, vtarget):
    n = torch.argmax(nout, -1)
    v = torch.argmax(vout, -1)
    acc = torch.sum((n == ntarget)*(v == vtarget)).float()/nout.shape[0]
    return acc

def accuracy(class_embedding, vis_trans_out, target, inds):
    inds = torch.from_numpy(inds).to(device)
    temp_vis = vis_trans_out.unsqueeze(1).expand(vis_trans_out.shape[0], class_embedding.shape[0], vis_trans_out.shape[1])
    temp_cemb = class_embedding.unsqueeze(0).expand(vis_trans_out.shape[0], class_embedding.shape[0], vis_trans_out.shape[1])
    preds = torch.argmax(torch.sum(temp_vis*temp_cemb, axis=2), axis = 1)
    acc = torch.sum(inds[preds] == target).item()/(preds.shape[0])
    # print(torch.sum(inds[preds] == target).item())
    return acc, preds

def ce_accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

def get_text_data(target, s2v_labels):
    return s2v_labels[target].view(target.shape[0], att_input_size)


def get_w2v_data(target_nouns, w2v_model, nouns):
    
    srt_inp = torch.zeros([target_nouns.shape[0], 1, target_nouns.shape[1]]).float()
    noun_inp = target_nouns.view(target_nouns.shape[0], 1, target_nouns.shape[1]).float()
#     verb_inp = torch.view(target_nouns.shape[0], 1, target_nouns.shape[1])
    inp = torch.cat([srt_inp, noun_inp], 1)
    return inp

def zsl_validate(val_loader, visae, attae, epoch):
    with torch.no_grad():
        losses = AverageMeter()
        acces = AverageMeter()
        ce_loss_vals = []
        ce_acc_vals = []
        visae.eval()
        attae.eval()
        class_embeddings = attae(unseen_s2v_labels.to(device).float())[2]
        for i, (emb, target) in enumerate(val_loader):
            emb = emb.to(device)
            vis_emb = torch.log(1 + emb)
        
            vis_hidden, vis_out, vis_trans_out = visae(vis_emb)
            vis_recons_loss = criterion2(vis_out, vis_emb)
            
            att_emb = get_text_data(target, s2v_labels.float()).to(device)
            att_hidden, att_out, att_trans_out = attae(att_emb)
            att_recons_loss = criterion2(att_out, att_emb)
            
            # mmd loss
            loss_mmd = MMDLoss(vis_hidden, att_hidden).to(device)
            
            
            # supervised binary prediction loss
            pred_loss = multi_class_hinge_loss(vis_trans_out, att_trans_out, target, class_embeddings)
       
            loss = pred_loss + vis_recons_loss + att_recons_loss + loss_mmd
            acc,_ = accuracy(class_embeddings, vis_trans_out, target.to(device), unseen_inds)
            losses.update(loss.item(), emb.size(0))
            acces.update(acc, emb.size(0))
            
            ce_loss_vals.append(loss.cpu().detach().numpy())
            ce_acc_vals.append(acc)
            if i % 20 == 0:
                print('ZSL Validation Epoch-{:<3d} {:3d}/{:3d} batches \t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                       epoch, i , len(val_loader), loss=losses, acc=acces))
                print('Vis Reconstruction Loss {:.4f}\t'
                      'Att Reconstruction Loss {:.4f}\t'
                      'MMD Loss {:.4f}\t'
                      'Supervised Binary Prediction Loss {:.4f}'.format(
                       vis_recons_loss.item(), att_recons_loss.item(), loss_mmd.item(), pred_loss.item()))
                
    return losses.avg, acces.avg


def train(train_loader, visae, attae, visae_optimizer, attae_optimizer, epoch):
    visae.train()
    attae.train()
    train_losses = AverageMeter()
    train_acces = AverageMeter()
    for i ,(emb, target) in enumerate(train_loader):
        emb = emb.to(device)
        with torch.no_grad():
            seen_class_embeddings = attae(seen_s2v_labels.to(device).float())[2]
            vis_emb = torch.log(1 + emb)
        
        vis_emb.retain_grad()
        vis_emb.requires_grad_(True)
        
        vis_hidden, vis_out, vis_trans_out = visae(vis_emb)
        vis_recons_loss = contractive_loss(vis_out, vis_emb, vis_hidden, criterion2, device, gamma=0.0001)
        
        att_emb = get_text_data(target, s2v_labels.float()).to(device)
        att_hidden, att_out, att_trans_out = attae(att_emb)
        att_recons_loss = criterion2(att_out, att_emb)
        
        # mmd loss
        loss_mmd = MMDLoss(vis_hidden, att_hidden).to(device)
        
        # supervised binary prediction loss
        pred_loss = multi_class_hinge_loss(vis_trans_out, att_trans_out, target, seen_class_embeddings)
    
        loss = pred_loss + vis_recons_loss + att_recons_loss + loss_mmd

        # backward
        visae_optimizer.zero_grad()  # clear gradients out before each mini-batch
        attae_optimizer.zero_grad()
        loss.backward()
        visae_optimizer.step()  # update parameters
        attae_optimizer.step() 
        
        acc, _ = accuracy(seen_class_embeddings, vis_trans_out.detach(), target.to(device), seen_inds)
        train_losses.update(loss.item(), emb.size(0))
        train_acces.update(acc, emb.size(0))
        if i%20 == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, len(train_loader), loss=train_losses, acc=train_acces))
            print('Vis Reconstruction Loss {:.4f}\t'
                'Att Reconstruction Loss {:.4f}\t'
                'MMD Loss {:.4f}\t'
                'Supervised Binary Prediction Loss {:.4f}'.format(
                vis_recons_loss.item(), att_recons_loss.item(), loss_mmd.item(), pred_loss.item()))
        
    return train_losses.avg, train_acces.avg


def validate(train_loader, visae, attae, epoch):
    visae.eval()
    attae.eval()
    train_losses = AverageMeter()
    train_acces = AverageMeter()
    for i ,(emb, target) in enumerate(train_loader):
        emb = emb.to(device)
        with torch.no_grad():
            seen_class_embeddings = attae(seen_s2v_labels.to(device).float())[2]
            vis_emb = torch.log(1 + emb)
        
        vis_emb.retain_grad()
        vis_emb.requires_grad_(True)
        
        vis_hidden, vis_out, vis_trans_out = visae(vis_emb)
        vis_recons_loss = contractive_loss(vis_out, vis_emb, vis_hidden, criterion2, device, gamma=0.0001)
        
        att_emb = get_text_data(target, s2v_labels.float()).to(device)
        att_hidden, att_out, att_trans_out = attae(att_emb)
        att_recons_loss = criterion2(att_out, att_emb)
        
        # mmd loss
        loss_mmd = MMDLoss(vis_hidden, att_hidden).to(device)
        
        # supervised binary prediction loss
        pred_loss = multi_class_hinge_loss(vis_trans_out, att_trans_out, target, seen_class_embeddings)
    
        loss = pred_loss + vis_recons_loss + att_recons_loss + loss_mmd

        acc, _ = accuracy(seen_class_embeddings, vis_trans_out.detach(), target.to(device), seen_inds)
        train_losses.update(loss.item(), emb.size(0))
        train_acces.update(acc, emb.size(0))
        if i%20 == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, len(train_loader), loss=train_losses, acc=train_acces))
            print('Vis Reconstruction Loss {:.4f}\t'
                'Att Reconstruction Loss {:.4f}\t'
                'MMD Loss {:.4f}\t'
                'Supervised Binary Prediction Loss {:.4f}'.format(
                vis_recons_loss.item(), att_recons_loss.item(), loss_mmd.item(), pred_loss.item()))
        
    return train_losses.avg, train_acces.avg

max_epochs = 300
gbest = 0
zbest = 0
if phase == 'val':
    zvisae_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/' + 'gvisae_best.pth.tar'
    zattae_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/' + 'gattae_best.pth.tar'
else:
    zvisae_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/' + 'main_gvisae_best.pth.tar'
    zattae_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/' + 'main_gattae_best.pth.tar'
early_stop = 0
print(visae)
print(attae)
for epoch in range(0, max_epochs):
    train_loss, train_acc = train(train_loader, visae, attae, visae_optimizer, attae_optimizer, epoch)
    zsl_loss, zsl_acc = zsl_validate(zsl_loader, visae, attae, epoch)
    val_loss, val_acc = validate(val_loader, visae, attae, epoch)
    print('Epoch-{:<3d} \tTrain loss {:.4f}\tTrain accuracy {:.4f}\tZValid: zloss {:.4f}\tZaccu {:.4f}\tValid: val_loss {:.4f}\tval_accu {:.4f}'
        .format(epoch + 1, train_loss, train_acc, zsl_loss, zsl_acc, val_loss, val_acc))
    

    if zsl_acc > zbest:
        print('Epoch %d: %s from %.4f to %.4f, '
                    'saving model to %s'
                    % (epoch, 'zsl_acc improved', zbest, zsl_acc, zvisae_checkpoint))
        zbest = zsl_acc 
        zbest_epoch = epoch
        save_checkpoint({ 'epoch': epoch,
            'revise_state_dict': visae.state_dict(),
            'best_loss': zsl_loss,
            'best_acc' : zsl_acc,
            'monitor': 'zsl_acc',
            'revise_optimizer': visae_optimizer.state_dict()
        }, zvisae_checkpoint)
        save_checkpoint({ 'epoch': epoch,
            'revise_state_dict': attae.state_dict(),
            'best_loss': zsl_loss,
            'best_acc' : zsl_acc,
            'monitor': 'zsl_acc',
            'revise_optimizer': attae_optimizer.state_dict()
        }, zattae_checkpoint)
    
    visae_scheduler.step(zsl_acc)
    attae_scheduler.step(zsl_acc)
        

print('Best %s: %.4f from epoch-%d' % ('zsl_acc', zbest, zbest_epoch))
