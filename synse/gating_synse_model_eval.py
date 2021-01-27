import argparse
import time
import shutil
import os
import os.path as osp
import csv
import numpy as np
import torch
import torch.nn as nn
from data_cnn60 import NTUDataLoaders, AverageMeter, make_dir, get_cases, get_num_classes

parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--st', type=str, help="split type")
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--wdir', type=str, help="directory to save weights path")
parser.add_argument('--le', type=str, help="language embedding model")
parser.add_argument('--ve', type=str, help="visual embedding model")
parser.add_argument('--phase', type=str, help="train or val")
parser.add_argument('--temp', type=int, help="temperature")
parser.add_argument('--ntu', type=int, help="num_classes")
parser.add_argument('--thresh', type=float, help="temperature")

args = parser.parse_args()

ss = args.ss
st = args.st
dataset = args.dataset
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_classes = args.ntu
temp = args.temp
thresh = args.thresh

# gpu = '0'
# ss = 12
# st = 'r'
# dataset_path = 'ntu_results/shift_12_r_1'
# # wdir = gating_our
# le = 'bert_large'
# ve = 'shift'
# phase = 'train'
# num_classes = 60

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
# train_loader = ntu_loaders.get_train_loader(1024, 8)
# zsl_loader = ntu_loaders.get_val_loader(1024, 8)
# val_loader = ntu_loaders.get_test_loader(1024, 8)
# zsl_out_loader = ntu_loaders.get_val_out_loader(1024, 8)
# val_out_loader = ntu_loaders.get_test_out_loader(1024, 8)
# train_size = ntu_loaders.get_train_size()
# zsl_size = ntu_loaders.get_val_size()
# val_size = ntu_loaders.get_test_size()
# print('Train on %d samples, validate on %d samples' % (train_size, zsl_size))



if phase == 'val':
    gzsl_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_classes - ss) +'.npy')
    unseen_inds = np.sort(np.load('../resources/label_splits/' + st + 'v' + str(ss) + '_0.npy'))
    seen_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_classes - ss -ss) + '_0.npy')
else:
    gzsl_inds = np.arange(num_classes)
    unseen_inds = np.sort(np.load('../resources/label_splits/' + st + 'u' + str(ss) + '.npy'))
    seen_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_classes - ss) + '.npy')


tars = np.load(dataset + '/g_label.npy')

test_y = []
for i in tars:
    if i in unseen_inds:
        test_y.append(0)
    else:
        test_y.append(1)

test_zs = np.load(wdir + '/' + le + '/synse_' + str(ss) + '_r_gzsl_zs.npy')
test_seen = np.load(dataset + '/gtest_out.npy')


def temp_scale(seen_features, T):
    return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (seen_features + 1e-12)/T])

prob_test_zs = test_zs
prob_test_seen = temp_scale(test_seen, temp)

feat_test_zs = np.sort(prob_test_zs, 1)[:,::-1][:,:ss]
feat_test_seen = np.sort(prob_test_seen, 1)[:,::-1][:,:ss]

gating_test_x = np.concatenate([feat_test_zs, feat_test_seen], 1)
gating_test_y = test_y


import pickle as pkl
with open(wdir + '/' + le + '/gating_model.pkl', 'rb') as f:
    gating_model = pkl.load(f)


prob_gate = gating_model.predict_proba(gating_test_x)
pred_test = 1 - prob_gate[:, 0]>thresh
np.sum(pred_test == test_y)/len(test_y)
a = prob_gate
b = np.zeros(prob_gate.shape[0])
p_gate_seen = prob_gate[:, 1]
prob_y_given_seen = prob_test_seen + (1/num_classes)*np.repeat((1 - p_gate_seen)[:, np.newaxis], num_classes, 1)
p_gate_unseen = prob_gate[:, 0]
prob_y_given_unseen = prob_test_zs + (1/ss)*np.repeat((1 - p_gate_unseen)[:, np.newaxis], ss, 1)
prob_seen = prob_y_given_seen*np.repeat(p_gate_seen[:, np.newaxis], num_classes, 1)
prob_unseen = prob_y_given_unseen*np.repeat(p_gate_unseen[:, np.newaxis], ss, 1)

final_preds = []
seen_count = 0
tot_seen = 0
unseen_count = 0
tot_unseen = 0
gseen_count = 0
gunseen_count = 0
for i in range(len(gating_test_y)):
    if pred_test[i] == 1:
        pred = seen_inds[np.argmax(prob_test_seen[i, seen_inds])]
    else:
        pred = unseen_inds[np.argmax(prob_test_zs[i, :])]
    
    if tars[i] in seen_inds:
        tot_seen += 1
        if pred_test[i] == 1:
            gseen_count += 1
        if pred == tars[i]:
            seen_count += 1
    else:
        if pred_test[i] == 0:
            gunseen_count+=1
        tot_unseen += 1
        if pred == tars[i]:
            unseen_count += 1
    final_preds.append(pred)


seen_acc = seen_count/tot_seen
print('seen_accuracy', seen_acc)
unseen_acc = unseen_count/tot_unseen
print('unseen_accuracy', unseen_acc)
h_mean = 2*seen_acc*unseen_acc/(seen_acc + unseen_acc)
print('h_mean', h_mean)

#softgating

# prob_seen[:,unseen_inds] = prob_unseen
# final_prob = prob_seen
# final_pred = np.argmax(final_prob, -1)

# seen_count = 0
# seen_hit = 0
# unseen_count = 0
# unseen_hit = 0
# gating_seen_hit = 0
# gating_unseen_hit = 0
# for i, gt in enumerate(tars):
#     if gt in seen_inds:
#         seen_count += 1
#         if final_pred[i] == gt:
#             seen_hit += 1
#     else:
#         if prob_gate[i, 0] > prob_gate[i, 1]:
#             gating_unseen_hit += 1
#         unseen_count += 1
#         if final_pred[i] == gt:
#             unseen_hit += 1

# seen_acc = seen_hit/seen_count
# unseen_acc = unseen_hit/unseen_count
# h_mean = 2*seen_acc*unseen_acc/(seen_acc + unseen_acc)