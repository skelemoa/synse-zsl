import argparse
import time
import shutil
import os
import os.path as osp
import csv
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import pickle as pkl


parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--st', type=str, help="split type")
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--wdir', type=str, help="directory to save weights path")
parser.add_argument('--le', type=str, help="language embedding model")
parser.add_argument('--ve', type=str, help="visual embedding model")
parser.add_argument('--phase', type=str, help="train or val")
parser.add_argument('--ntu', type=str, help="number of classes")
args = parser.parse_args()

ss = args.ss
st = args.st
dataset = args.dataset
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_classes = args.ntu

# gpu = '0'
# ss = 5
# st = 'r'
# dataset_path = 'ntu_results/shift_val_5_r'
# wdir = 'pos_aware_cada_vae_concatenated_latent_space_shift_5_r_val'
# le = 'bert_large'
# ve = 'shift'
# phase = 'val'
# num_classes = 60

seed = 5
np.random.seed(seed)
    
def temp_scale(seen_features, T):
    return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (seen_features + 1e-12)/T])


unseen_zs = np.load('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/synse_' + str(ss) + '_r_unseen_zs.npy')
seen_zs = np.load('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/synse_' + str(ss) + '_r_seen_zs.npy')
unseen_train = np.load('/ssd_scratch/cvit/pranay.gupta/' + dataset +'/ztest_out.npy')
seen_train = np.load('/ssd_scratch/cvit/pranay.gupta/' + dataset + '/val_out.npy')


for f in [ss]:
    best_model = None
    best_acc = 0
    best_thresh = 0
    for t in range(1, 10):
        fin_val_acc = 0
        fin_train_acc = 0
        prob_unseen_zs = unseen_zs
        prob_unseen_train = temp_scale(unseen_train, t)
        prob_seen_zs = seen_zs
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
        pred_train = model.predict(gating_train_x)
        train_acc = np.sum(pred_train == gating_train_y)/len(gating_train_y)
        fin_train_acc += train_acc
        
        if fin_val_acc > best_acc:
            best_temp = t
            best_acc = fin_val_acc
            best_thresh = bestT
            best_model = model
    print('best validation accuracy for the gating model', best_acc)
    print('best threshold', best_thresh)
    print('best temperature', best_temp)

with open('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir.replace('_val', '') + '/' + le + '/gating_model.pkl', 'wb') as f:
    pkl.dump(best_model, f)
    f.close()
