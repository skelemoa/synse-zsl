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
from data_cnn60 import NTUDataLoaders, AverageMeter, make_dir, get_cases, get_num_classes
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import torch.nn.functional as F
from alignment_module import AE, PosMmen, JPosMmen, MMDLoss, multi_class_hinge_loss, triplet_loss, Mmen
import pickle as pkl


parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--st', type=str, help="split type")
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

# gpu = '0'
# ss = 10
# st = 'r'
# dataset_path = 'ntu_results/shift_val_' + str(ss) + '_r'
# # wdir = gating_our
# le = 'bert'
# ve = 'shift'
# phase = 'val'
# num_classes = 120

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda")
print(torch.cuda.device_count())


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

npm_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/gvacnn_NPM.pth.tar'
vpm_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/gvacnn_VPM.pth.tar'
jm_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/gvacnn_JM.pth.tar'
NounPosMmen = Mmen(vis_emb_input_size, noun_emb_input_size, output_size).to(device)
NounPosMmen.load_state_dict(torch.load(npm_checkpoint)['npm_state_dict'], strict=False)
NounPosMmen_optimizer = optim.Adam(NounPosMmen.parameters(), lr=0.0001)
NounPosMmen_scheduler = ReduceLROnPlateau(NounPosMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)


VerbPosMmen = Mmen(vis_emb_input_size, verb_emb_input_size, output_size).to(device)
VerbPosMmen.load_state_dict(torch.load(vpm_checkpoint)['vpm_state_dict'], strict=False)
VerbPosMmen_optimizer = optim.Adam(VerbPosMmen.parameters(), lr=0.0001)
VerbPosMmen_scheduler = ReduceLROnPlateau(VerbPosMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)


joint_emb_size = 100
joint_hidden_size = 50
JointMmen = JPosMmen(joint_emb_size, joint_hidden_size).to(device)
JointMmen.load_state_dict(torch.load(jm_checkpoint)['vpm_state_dict'], strict=False)
JointMmen_optimizer = optim.Adam(JointMmen.parameters(), lr=0.0001, weight_decay = 0.001)
JointMmen_scheduler = ReduceLROnPlateau(JointMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_train_loader(1024, 8)
zsl_loader = ntu_loaders.get_val_loader(1024, 8)
val_loader = ntu_loaders.get_test_loader(1024, 8)
zsl_out_loader = ntu_loaders.get_val_out_loader(1024, 8)
val_out_loader = ntu_loaders.get_test_out_loader(1024, 8)
train_size = ntu_loaders.get_train_size()
zsl_size = ntu_loaders.get_val_size()
val_size = ntu_loaders.get_test_size()
print('Train on %d samples, validate on %d samples' % (train_size, zsl_size))



nouns_vocab = np.load('../resources/nouns_vocab.npy')
verbs_vocab = np.load('../resources/verbs_vocab.npy')
nouns = nouns_vocab[np.argmax(np.load('../resources/nouns_ohe.npy'), -1)][:num_classes]
verbs = verbs_vocab[np.argmax(np.load('../resources/verbs_ohe.npy'), -1)][:num_classes]
labels = np.load('../resources/labels.npy')

if phase == 'val':
    gzsl_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_classes - ss) +'.npy')
    unseen_inds = np.sort(np.load('../resources/label_splits/' + st + 'v' + str(ss) + '_0.npy'))
    seen_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_classes - ss -ss) + '_0.npy')
else:
    gzsl_inds = np.arange(num_classes)
    unseen_inds = np.sort(np.load('../resources/label_splits/' + st + 'u' + str(ss) + '.npy'))
    seen_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_classes - ss) + '.npy')

unseen_labels = labels[unseen_inds]
seen_labels = labels[seen_inds]

unseen_nouns = nouns[unseen_inds]
unseen_verbs = verbs[unseen_inds]
seen_nouns = nouns[seen_inds]
seen_verbs = verbs[seen_inds]
verb_corp = np.unique(verbs[gzsl_inds])
noun_corp = np.unique(nouns[gzsl_inds])

verb_emb = torch.from_numpy(np.load('../resources/ntu' + str(num_classes) + '_' + le + '_verb.npy')).view([num_classes, verb_emb_input_size])
verb_emb = verb_emb/torch.norm(verb_emb, dim = 1).view([num_classes, 1]).repeat([1, verb_emb_input_size])
noun_emb = torch.from_numpy(np.load('../resources/ntu' + str(num_classes) + '_' + le + '_noun.npy')).view([num_classes, noun_emb_input_size])
noun_emb = noun_emb/torch.norm(noun_emb, dim = 1).view([num_classes, 1]).repeat([1, noun_emb_input_size])

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
    return acc, torch.sum(temp_vis*temp_cemb, axis=2)

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
        scores = []
        noun_scores = []
        verb_scores = []
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
            
            vis_noun_transform,  noun_transform = NounPosMmen(emb, noun_emb_target.to(device).float())
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
            ce_acc, score = accuracy(fin_text_embedding, fin_vis, target.to(device), unseen_inds)
            losses.update(loss.item(), inputs.size(0))
            acces.update(ce_acc, inputs.size(0))
            tars.append(target)
            scores.append(score)
            ce_loss_vals.append(loss.cpu().detach().numpy())
            ce_acc_vals.append(ce_acc)

            if i % 20 == 0:
                print('Epoch-{:<3d} {:3d}/{:3d} batches \t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, i + 1, len(val_loader), loss=losses, acc=acces))
        return losses.avg, acces.avg, scores

def validate(train_loader, epoch, margin):

    losses = AverageMeter()
    acces = AverageMeter()
    ce_loss_vals = []
    ce_acc_vals = []
    scores = []
    for i, (inputs, target) in enumerate(train_loader):
        # (inputs, target) = next(iter(train_loader))
        emb = inputs.to(device)
        scores.append(emb)
    return scores

zsl_loss, zsl_acc, unseen_zs = zsl_validate(zsl_loader, 0, 0.3)
seen_train = validate(val_out_loader, 0, 0.3)
zsl_loss, zsl_acc, seen_zs = zsl_validate(val_loader, 0, 0.3)
unseen_train = validate(zsl_out_loader, 0, 0.3)

unseen_zs = np.array([j.cpu().detach().numpy() for i in unseen_zs for j in i])
unseen_train = np.array([j.cpu().detach().numpy() for i in unseen_train for j in i])
seen_zs = np.array([j.cpu().detach().numpy() for i in seen_zs for j in i])
seen_train = np.array([j.cpu().detach().numpy() for i in seen_train for j in i])

from sklearn.linear_model import LogisticRegression

def temp_scale(seen_features, T):
    return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (seen_features + 1e-12)/T])

for f in [ss]:
    best_temp = 0
    best_acc = 0
    best_model = None
    best_thresh = 0
    for t in range(2, 10):
        fin_val_acc = 0
        fin_train_acc = 0
        prob_unseen_zs = np.array([np.exp(i)/np.sum(np.exp(i)) for i in unseen_zs])
        prob_unseen_train = temp_scale(unseen_train, t)
        prob_seen_zs = np.array([np.exp(i)/np.sum(np.exp(i)) for i in seen_zs])
        prob_seen_train = temp_scale(seen_train, t)

        feat_unseen_zs = np.sort(prob_unseen_zs, 1)[:,::-1][:,:f]
        feat_unseen_train = np.sort(prob_unseen_train, 1)[:,::-1][:,:f]
        feat_seen_zs = np.sort(prob_seen_zs, 1)[:,::-1][:,:f]
        feat_seen_train = np.sort(prob_seen_train, 1)[:,::-1][:,:f]

        val_unseen_inds = np.random.choice(np.arange(feat_unseen_train.shape[0]), 300, replace=False)
        val_seen_inds = np.random.choice(np.arange(feat_seen_train.shape[0]), 400, replace=False)
        train_unseen_inds = np.array(list(set(list(np.arange(feat_unseen_train.shape[0]))) - set(list(val_unseen_inds))))
        train_seen_inds = np.array(list(set(list(np.arange(feat_seen_train.shape[0]))) - set(list(val_seen_inds))))

        gating_train_x = np.concatenate([np.concatenate([feat_unseen_zs[train_unseen_inds, :], feat_unseen_train[train_unseen_inds, :]], 1), np.concatenate([feat_seen_zs[train_seen_inds, :], feat_seen_train[train_seen_inds, :]], 1)], 0)
        gating_train_y = [0]*len(train_unseen_inds) + [1]*len(train_seen_inds)
        gating_val_x = np.concatenate([np.concatenate([feat_unseen_zs[val_unseen_inds, :], feat_unseen_train[val_unseen_inds, :]], 1), np.concatenate([feat_seen_zs[val_seen_inds, :], feat_seen_train[val_seen_inds, :]], 1)], 0)
        gating_val_y = [0]*len(val_unseen_inds) + [1]*len(val_seen_inds)

        train_inds = np.arange(gating_train_x.shape[0])
        np.random.shuffle(train_inds)
        model = LogisticRegression(random_state=0, C=1, solver='lbfgs', n_jobs=-1,
                                     multi_class='multinomial', verbose=0, max_iter=5000,
                                     ).fit(gating_train_x[train_inds, :], np.array(gating_train_y)[train_inds])
        prob = model.predict_proba(gating_val_x)
        best = 0
        bestT = 0
        for th in range(25, 75, 1):
            y = prob[:, 0] > th/100
            acc = np.sum((1 - y) == gating_val_y)/len(gating_val_y)
            if acc > best:
                best = acc
                bestT = th/100
        fin_val_acc += best
        if fin_val_acc > best_acc:
            best_temp = t
            best_acc = fin_val_acc
            best_thresh = bestT
            best_model = model
    print('best validation accuracy for the gating model', best_acc)
    print('best threshold', best_thresh)
    print('best temperature', best_temp)
        

with open('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/gating_model.pkl', 'wb') as f:
    pkl.dump(best_model, f)
    f.close()