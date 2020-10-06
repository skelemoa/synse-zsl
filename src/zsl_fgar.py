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
from alignment_module import JPosMmen, MMDLoss, multi_class_hinge_loss, triplet_loss, Mmen

parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--st', type=str, help="split type")
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--wdir', type=str, help="directory to save weights path")
parser.add_argument('--le', type=str, help="language embedding model")
parser.add_argument('--ve', type=str, help="visual embedding model")
parser.add_argument('--phase', type=str, help="train or val")
parser.add_argument('--gpu', type=str, help="gpu device number")
parser.add_argument('--ntu', type=int, help="ntu120 or ntu60")
args = parser.parse_args()

gpu = args.gpu
ss = args.ss
st = args.st
dataset_path = args.dataset
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_class = args.ntu

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda")
print(torch.cuda.device_count())


if not os.path.exists('../language_modelling/' + wdir):
    os.mkdir('../language_modelling/' + wdir)
if not os.path.exists('../language_modelling/' + wdir + '/' + le):
    os.mkdir('../language_modelling/' + wdir + '/' + le)

criterion2 = nn.MSELoss().to(device)

if ve == 'vacnn':
    vis_emb_input_size = 2048
elif ve == 'shift':
    vis_emb_input_size = 256
else: 
    pass    
    
text_hidden_size = 100
vis_hidden_size = 512
output_size = 50

if le == 'bert':
    noun_emb_input_size = 1024
    verb_emb_input_size = 1024
elif le == 'w2v':
    noun_emb_input_size = 300
    verb_emb_input_size = 300
else:
    pass

NounPosMmen = Mmen(vis_emb_input_size, noun_emb_input_size, output_size).to(device)
# NounPosMmen.load_state_dict(torch.load(npm_checkpoint)['npm_state_dict'], strict=False)
NounPosMmen_optimizer = optim.Adam(NounPosMmen.parameters(), lr=0.0001)
NounPosMmen_scheduler = ReduceLROnPlateau(NounPosMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)


VerbPosMmen = Mmen(vis_emb_input_size, verb_emb_input_size, output_size).to(device)
# VerbPosMmen.load_state_dict(torch.load(vpm_checkpoint)['vpm_state_dict'], strict=False)
VerbPosMmen_optimizer = optim.Adam(VerbPosMmen.parameters(), lr=0.0001)
VerbPosMmen_scheduler = ReduceLROnPlateau(VerbPosMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)

joint_emb_size = 100
joint_hidden_size = 50
JointMmen = JPosMmen(joint_emb_size, joint_hidden_size).to(device)
# JointMmen.load_state_dict(torch.load(jpm_checkpoint)['vpm_state_dict'], strict=False)
JointMmen_optimizer = optim.Adam(JointMmen.parameters(), lr=0.0001, weight_decay = 0.001)
JointMmen_scheduler = ReduceLROnPlateau(JointMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_train_loader(1024, 8)
zsl_loader = ntu_loaders.get_val_loader(1024, 8)
val_loader = ntu_loaders.get_test_loader(1024, 8)
train_size = ntu_loaders.get_train_size()
zsl_size = ntu_loaders.get_val_size()
val_size = ntu_loaders.get_test_size()
print('Train on %d samples, validate on %d samples' % (train_size, zsl_size))

nouns_vocab = np.load('../language_data/nouns_vocab.npy')
verbs_vocab = np.load('../language_data/verbs_vocab.npy')
nouns = nouns_vocab[np.argmax(np.load('../language_data/nouns_ohe.npy'), -1)][:num_class]
verbs = verbs_vocab[np.argmax(np.load('../language_data/verbs_ohe.npy'), -1)][:num_class]
labels = np.load('../language_data/labels.npy')

if phase == 'val':
    gzsl_inds = np.load('../label_splits/'+ st + 's' + str(num_class - ss) +'.npy')
    unseen_inds = np.sort(np.load('../label_splits/' + st + 'v' + str(ss) + '_0.npy'))
    seen_inds = np.load('../label_splits/'+ st + 's' + str(num_class -ss - ss) + '_0.npy')
else:
    gzsl_inds = np.arange(num_class)
    unseen_inds = np.sort(np.load('../label_splits/' + st + 'u' + str(ss) + '.npy'))
    seen_inds = np.load('../label_splits/'+ st + 's' + str(num_class  -ss) + '.npy')

unseen_labels = labels[unseen_inds]
seen_labels = labels[seen_inds]

unseen_nouns = nouns[unseen_inds]
unseen_verbs = verbs[unseen_inds]
seen_nouns = nouns[seen_inds]
seen_verbs = verbs[seen_inds]
verb_corp = np.unique(verbs[gzsl_inds])
noun_corp = np.unique(nouns[gzsl_inds])

verb_emb = torch.from_numpy(np.load('../language_data/' + le + '_verb.npy')[:num_class, :]).view([num_class, verb_emb_input_size])
verb_emb = verb_emb/torch.norm(verb_emb, dim = 1).view([num_class, 1]).repeat([1, verb_emb_input_size])
noun_emb = torch.from_numpy(np.load('../language_data/' + le + '_noun.npy')[:num_class, :]).view([num_class, noun_emb_input_size])
noun_emb = noun_emb/torch.norm(noun_emb, dim = 1).view([num_class, 1]).repeat([1, noun_emb_input_size])

unseen_verb_emb = verb_emb[unseen_inds, :]
unseen_noun_emb = noun_emb[unseen_inds, :]
seen_verb_emb = verb_emb[seen_inds, :]
seen_noun_emb = noun_emb[seen_inds, :]
print("loaded language embeddings")


def get_text_data(target, verb_emb, noun_emb):
    return verb_emb[target].view(target.shape[0], verb_emb_input_size).float(), noun_emb[target].view(target.shape[0], verb_emb_input_size).float()


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def accuracy(class_embedding, vis_trans_out, target, inds):
    inds = torch.from_numpy(inds).to(device)
    temp_vis = vis_trans_out.unsqueeze(1).expand(vis_trans_out.shape[0], class_embedding.shape[0], vis_trans_out.shape[1])
    temp_cemb = class_embedding.unsqueeze(0).expand(vis_trans_out.shape[0], class_embedding.shape[0], vis_trans_out.shape[1])
    preds = torch.argmax(torch.sum(temp_vis*temp_cemb, axis=2), axis = 1)
    acc = torch.sum(inds[preds] == target).item()/(preds.shape[0])
    return acc, inds[preds]


def train(train_loader, epoch, margin):

    losses = AverageMeter()
    acces = AverageMeter()
    ce_loss_vals = []
    ce_acc_vals = []
    NounPosMmen.train()
    VerbPosMmen.train()
    JointMmen.train()
    for i, (inputs, target) in enumerate(train_loader):
        emb = inputs.to(device)
        verb_emb_target, noun_emb_target = get_text_data(target, verb_emb, noun_emb)
        with torch.no_grad():
            verb_tar = []
            for p in verbs[target]:
                verb_tar.append((np.argwhere(verb_corp == p)[0][0]))

            noun_tar = []
            for p in nouns[target]:
                noun_tar.append((np.argwhere(noun_corp == p)[0][0]))


            emb = emb/torch.norm(emb, dim = 1).view([emb.size(0), 1]).repeat([1, emb.shape[1]])
            verb_embeddings = VerbPosMmen.TextArch(seen_verb_emb.to(device).float())
            noun_embeddings = NounPosMmen.TextArch(seen_noun_emb.to(device).float())
            noun_embeddings = noun_embeddings/torch.norm(noun_embeddings, dim = 1).view([noun_embeddings.size(0), 1]).repeat([1, noun_embeddings.shape[1]])
            verb_embeddings = verb_embeddings/torch.norm(verb_embeddings, dim = 1).view([verb_embeddings.size(0), 1]).repeat([1, verb_embeddings.shape[1]])
            joint_text_embedding = torch.cat([verb_embeddings, noun_embeddings], axis=1)
            fin_text_embedding = JointMmen.TextArch(joint_text_embedding)
            fin_text_embedding = fin_text_embedding/torch.norm(fin_text_embedding, dim = 1).view([fin_text_embedding.size(0), 1]).repeat([1, fin_text_embedding.shape[1]])
            
        vis_verb_transform, verb_transform = VerbPosMmen(emb, verb_emb_target.to(device).float())
        vis_verb_transform = vis_verb_transform/torch.norm(vis_verb_transform, dim = 1).view([vis_verb_transform.size(0), 1]).repeat([1, vis_verb_transform.shape[1]])
        verb_transform = verb_transform/torch.norm(verb_transform, dim = 1).view([verb_transform.size(0), 1]).repeat([1, verb_transform.shape[1]])
        verb_vis_loss = multi_class_hinge_loss(vis_verb_transform, verb_transform, target, verb_embeddings, margin).to(device)
        verb_verb_loss = triplet_loss(vis_verb_transform, torch.tensor(verb_tar), device, margin).to(device)


        vis_noun_transform, noun_transform = NounPosMmen(emb, noun_emb_target.to(device).float())
        vis_noun_transform = vis_noun_transform/torch.norm(vis_noun_transform, dim = 1).view([vis_noun_transform.size(0), 1]).repeat([1, vis_noun_transform.shape[1]])
        noun_transform = noun_transform/torch.norm(noun_transform, dim = 1).view([noun_transform.size(0), 1]).repeat([1, noun_transform.shape[1]])
        noun_vis_loss = multi_class_hinge_loss(vis_noun_transform, noun_transform, target, noun_embeddings, margin).to(device)
        noun_noun_loss = triplet_loss(vis_noun_transform, torch.tensor(noun_tar), device, margin).to(device)    


        joint_vis = torch.cat([vis_verb_transform, vis_noun_transform], axis = 1)
        joint_text = torch.cat([verb_transform,  noun_transform], axis = 1)
        fin_vis, fin_text = JointMmen(joint_vis, joint_text)
        fin_text = fin_text/torch.norm(fin_text, dim = 1).view([fin_text.size(0), 1]).repeat([1, fin_text.shape[1]])
        fin_vis = fin_vis/torch.norm(fin_vis, dim = 1).view([fin_vis.size(0), 1]).repeat([1, fin_vis.shape[1]])
        joint_loss = multi_class_hinge_loss(fin_vis, fin_text, target, fin_text_embedding, margin).to(device)
        
        loss = verb_vis_loss + noun_vis_loss 
        loss += 0.01*(verb_verb_loss) + 0.01*(noun_noun_loss)
        loss += joint_loss

        # backward
        NounPosMmen_optimizer.zero_grad()
        VerbPosMmen_optimizer.zero_grad()
        JointMmen_optimizer.zero_grad()
        # PrpPosMmen_optimizer.zero_grad()
        loss.backward()
        NounPosMmen_optimizer.step()
        VerbPosMmen_optimizer.step()
        JointMmen_optimizer.step()

        # ce acc
        ce_acc, _ = accuracy(fin_text_embedding, fin_vis, target.to(device), seen_inds)
        losses.update(loss.item(), inputs.size(0))
        acces.update(ce_acc, inputs.size(0))

        ce_loss_vals.append(loss.cpu().detach().numpy())
        ce_acc_vals.append(ce_acc)
        if i % 20 == 0:
            print('Epoch-{:<3d} \t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, loss=losses, acc=acces))
            # print('vvrl {:.4f}\t'
            #         'nnrl {:.4f}\t'.format(vis_verb_reconstruction.item(), vis_noun_reconstruction.item()))
            # print('vvvl {:.4f}\t'
            #         'nntl {:.4f}\t'.format(verb_verb_loss.item(), noun_noun_loss.item()))
            # print('vtvl {:.4f}\t'
            #         'vtnl {:.4f}\t'.format(verb_vis_loss.item(), noun_vis_loss.item()))
            # print('vvmmdl {:.4f}\t'
            #         'vnmmdl {:.4f}\t'.format(verb_vis_mmd_loss.item(), noun_vis_mmd_loss.item())) 
            # print('joint loss {:.4f}\t'.format(joint_loss.item()))
    return losses.avg, acces.avg


def zsl_validate(val_loader, epoch, margin):

    with torch.no_grad():
        losses = AverageMeter()
        acces = AverageMeter()
        ce_loss_vals = []
        ce_acc_vals = []
        NounPosMmen.eval()
        VerbPosMmen.eval()
        JointMmen.eval()
        tars = []
        preds = []
        for i, (inputs, target) in enumerate(val_loader):
            emb = inputs.to(device)
            verb_emb_target, noun_emb_target = get_text_data(target, verb_emb, noun_emb)
            verb_tar = []
            for p in verbs[target]:
                verb_tar.append(np.argwhere(verb_corp == p)[0][0])

            noun_tar = []
            for p in np.unique(nouns[target]):
                noun_tar.append(np.argwhere(noun_corp == p)[0][0])

            emb = emb/torch.norm(emb, dim = 1).view([emb.size(0), 1]).repeat([1, emb.shape[1]])
            verb_embeddings = VerbPosMmen.TextArch(unseen_verb_emb.to(device).float())
            noun_embeddings = NounPosMmen.TextArch(unseen_noun_emb.to(device).float())
            noun_embeddings = noun_embeddings/torch.norm(noun_embeddings, dim = 1).view([noun_embeddings.size(0), 1]).repeat([1, noun_embeddings.shape[1]])
            verb_embeddings = verb_embeddings/torch.norm(verb_embeddings, dim = 1).view([verb_embeddings.size(0), 1]).repeat([1, verb_embeddings.shape[1]])
            joint_text_embedding = torch.cat([verb_embeddings, noun_embeddings], axis=1)
            fin_text_embedding = JointMmen.TextArch(joint_text_embedding)
            fin_text_embedding = fin_text_embedding/torch.norm(fin_text_embedding, dim = 1).view([fin_text_embedding.size(0), 1]).repeat([1, fin_text_embedding.shape[1]])
            
            vis_verb_transform, verb_transform = VerbPosMmen(emb, verb_emb_target.to(device).float())
            vis_verb_transform = vis_verb_transform/torch.norm(vis_verb_transform, dim = 1).view([vis_verb_transform.size(0), 1]).repeat([1, vis_verb_transform.shape[1]])
            verb_transform = verb_transform/torch.norm(verb_transform, dim = 1).view([verb_transform.size(0), 1]).repeat([1, verb_transform.shape[1]])
            verb_vis_loss = multi_class_hinge_loss(vis_verb_transform, verb_transform, target, verb_embeddings, margin).to(device)
            verb_verb_loss = triplet_loss(vis_verb_transform, torch.tensor(verb_tar), device, margin).to(device)
            
            vis_noun_transform, noun_transform = NounPosMmen(emb, noun_emb_target.to(device).float())
            vis_noun_transform = vis_noun_transform/torch.norm(vis_noun_transform, dim = 1).view([vis_noun_transform.size(0), 1]).repeat([1, vis_noun_transform.shape[1]])
            noun_transform = noun_transform/torch.norm(noun_transform, dim = 1).view([noun_transform.size(0), 1]).repeat([1, noun_transform.shape[1]])
            noun_vis_loss = multi_class_hinge_loss(vis_noun_transform, noun_transform, target, noun_embeddings, margin).to(device)
            noun_noun_loss = triplet_loss(vis_noun_transform, torch.tensor(noun_tar), device, margin).to(device)

        
            joint_vis = torch.cat([vis_verb_transform,  vis_noun_transform], axis = 1)
            joint_text = torch.cat([verb_transform,  noun_transform], axis = 1)
            fin_vis, fin_text = JointMmen(joint_vis, joint_text)
            fin_text = fin_text/torch.norm(fin_text, dim = 1).view([fin_text.size(0), 1]).repeat([1, fin_text.shape[1]])
            fin_vis = fin_vis/torch.norm(fin_vis, dim = 1).view([fin_vis.size(0), 1]).repeat([1, fin_vis.shape[1]])
            joint_loss = multi_class_hinge_loss(fin_vis, fin_text, target, fin_text_embedding, margin).to(device)
            
        
            loss = verb_vis_loss + noun_vis_loss
            loss += 0.01*(verb_verb_loss) + 0.01*(noun_noun_loss)
            loss += joint_loss
            
            # ce acc
            ce_acc, pred = accuracy(fin_text_embedding, fin_vis, target.to(device), unseen_inds)
            losses.update(loss.item(), inputs.size(0))
            acces.update(ce_acc, inputs.size(0))
            preds.append(pred)
            tars.append(target)
            ce_loss_vals.append(loss.cpu().detach().numpy())
            ce_acc_vals.append(ce_acc)

            if i % 20 == 0:
                print('Epoch-{:<3d} {:3d}/{:3d} batches \t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, i + 1, len(val_loader), loss=losses, acc=acces))
            #     print('vvrl {:.4f}\t'
            #           'nnrl {:.4f}\t'.format(vis_verb_reconstruction.item(), vis_noun_reconstruction.item()))
            #     print('vvvl {:.4f}\t'
            #           'nntl {:.4f}\t'.format(verb_verb_loss.item(), noun_noun_loss.item()))
            #     print('vtvl {:.4f}\t'
            #           'vtnl {:.4f}\t'.format(verb_vis_loss.item(), noun_vis_loss.item()))
            #     print('vvmmdl {:.4f}\t'
            #           'vnmmdl {:.4f}\t'.format(verb_vis_mmd_loss.item(), noun_vis_mmd_loss.item()))
                # print('joint loss {:.4f}'.format(joint_loss.item()))
        return losses.avg, acces.avg, preds, tars


max_epochs = 2000
best = 0
best_epoch = 0
zbest = 0
margin = 0.3
train_loss = 0
train_acc = 0
zsl_loss = 0
zsl_acc = 0
early_stop = 0
val_loss = 0
val_acc = 0

print(VerbPosMmen)
print(NounPosMmen)
if phase == 'val':
    npm_zcheckpoint = '../language_modelling/' + wdir + '/' + le + '/' + 'gvacnn_NPM.pth.tar'
    vpm_zcheckpoint = '../language_modelling/' + wdir + '/' + le + '/' + 'gvacnn_VPM.pth.tar'
    jm_zcheckpoint = '../language_modelling/'+ wdir + '/' + le + '/' + 'gvacnn_JM.pth.tar'
else:
    npm_zcheckpoint = '../language_modelling/' + wdir + '/' + le + '/' + 'main_gvacnn_NPM.pth.tar'
    vpm_zcheckpoint = '../language_modelling/' + wdir + '/' + le + '/' + 'main_gvacnn_VPM.pth.tar'
    jm_zcheckpoint = '../language_modelling/'+ wdir + '/' + le + '/' + 'main_gvacnn_JM.pth.tar'
for epoch in range(0, max_epochs):
        t_start = time.time()
        train_loss, train_acc = train(train_loader, epoch, margin)        
        zsl_loss, zsl_acc = zsl_validate(zsl_loader, epoch, margin)
        
        print('Epoch-{:<3d} {:.1f}s\t'
            'Train: loss {:.4f}\taccu {:.4f}\tZValid: zloss {:.4f}\tZaccu {:.4f}'
            .format(epoch + 1, time.time() - t_start, train_loss, train_acc, zsl_loss, zsl_acc))
        
        current = zsl_acc
        if np.greater(current, best):
            print('Epoch %d: %sd from %.4f to %.4f, '
                'saving model to %s'
                % (epoch , 'zsl_acc improved', best, current, npm_zcheckpoint))
            best = current
            best_epoch = epoch
            save_checkpoint({ 'epoch': epoch + 1,
                'npm_state_dict': NounPosMmen.state_dict(),
                'best_loss': zsl_loss,
                'best_acc' : zsl_acc,
                'monitor': 'gzsl_acc',
                'npm_optimizer': NounPosMmen_optimizer.state_dict()
            }, npm_zcheckpoint)
            save_checkpoint({ 'epoch': epoch + 1,
                'vpm_state_dict': VerbPosMmen.state_dict(),
                'best_loss': zsl_loss,
                'best_acc' : zsl_acc,
                'monitor': 'gzsl_acc',
                'vpm_optimizer': VerbPosMmen_optimizer.state_dict()
            }, vpm_zcheckpoint)
            save_checkpoint({ 'epoch': epoch + 1,
                'vpm_state_dict': JointMmen.state_dict(),
                'best_loss': zsl_loss,
                'best_acc' : zsl_acc,
                'monitor': 'gzsl_acc',
                'vpm_optimizer': JointMmen_optimizer.state_dict()
            }, jm_zcheckpoint)
            early_stop = 0
        else:
            early_stop += 1
            print('Epoch %d: %s did not %s' % (epoch + 1, 'zsl_acc', 'improve'))
        
        NounPosMmen_scheduler.step(zsl_acc)
        VerbPosMmen_scheduler.step(zsl_acc)
        JointMmen_scheduler.step(zsl_acc)
        if early_stop > 40:
            print('early stopping')
            break

print('Best %s: %.4f from epoch-%d' % ('zsl_acc', best, best_epoch))