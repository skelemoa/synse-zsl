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
from cada_vae import Encoder, Decoder, KL_divergence, Wasserstein_distance, reparameterize, triplet_loss, MLP


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
parser.add_argument('--num_cycles', type=int, help="no of cycles")
parser.add_argument('--num_epoch_per_cycle', type=int, help="number_of_epochs_per_cycle")
parser.add_argument('--latent_size', type=int, help="Latent dimension")
parser.add_argument('--mode', type=str, help="Mode")
parser.add_argument('--load_epoch', type=int, help="load epoch", default=None)
parser.add_argument('--load_classifier', action='store_true')
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
num_cycles = args.num_cycles
cycle_length = args.num_epoch_per_cycle
latent_size = args.latent_size
load_epoch = args.load_epoch
mode = args.mode
load_classifier = args.load_classifier

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

if ve == 'vacnn':
    vis_emb_input_size = 2048
elif ve == 'shift':
    vis_emb_input_size = 256
elif ve == 'msg3d':
    vis_emb_input_size = 384
else: 
    pass    
    
text_hidden_size = 100
vis_hidden_size = 100

if le == 'bert':
    text_emb_input_size = 1024
    # verb_emb_input_size = 1024
elif le == 'w2v':
    text_emb_input_size = 300
    # verb_emb_input_size = 300
else:
    pass

sequence_encoder = Encoder([vis_emb_input_size, latent_size]).to(device)
sequence_decoder = Decoder([latent_size,  vis_emb_input_size]).to(device)
text_encoder = Encoder([text_emb_input_size, latent_size]).to(device)
text_decoder = Decoder([latent_size, text_emb_input_size]).to(device)

params = []
for model in [sequence_encoder, sequence_decoder, text_encoder, text_decoder]:
    params += list(model.parameters())

optimizer = optim.Adam(params, lr = 0.0001)
# NounPosMmen_scheduler = ReduceLROnPlateau(NounPosMmen_optimizer, mode='max', factor=0.1, patience=14, cooldown=6, verbose=True)

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_train_loader(64, 8)
zsl_loader = ntu_loaders.get_val_loader(64, 8)
val_loader = ntu_loaders.get_test_loader(64, 8)
train_size = ntu_loaders.get_train_size()
zsl_size = ntu_loaders.get_val_size()
val_size = ntu_loaders.get_test_size()
print('Train on %d samples, validate on %d samples' % (train_size, val_size))


labels = np.load('../resources/labels.npy')

if phase == 'val':
    gzsl_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_class - ss) +'.npy')
    unseen_inds = np.sort(np.load('../resources/label_splits/' + st + 'v' + str(ss) + '_0.npy'))
    seen_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_class -ss - ss) + '_0.npy')
else:
    gzsl_inds = np.arange(num_class)
    unseen_inds = np.sort(np.load('../resources/label_splits/' + st + 'u' + str(ss) + '.npy'))
    seen_inds = np.load('../resources/label_splits/'+ st + 's' + str(num_class  -ss) + '.npy')

unseen_labels = labels[unseen_inds]
seen_labels = labels[seen_inds]

labels_emb = torch.from_numpy(np.load('../resources/ntu' + str(num_class) + '_' + le + '_labels.npy')).view([num_class, text_emb_input_size])
labels_emb = labels_emb/torch.norm(labels_emb, dim = 1).view([num_class, 1]).repeat([1, text_emb_input_size])

unseen_labels_emb = labels_emb[unseen_inds, :]
seen_labels_emb = labels_emb[seen_inds, :]
print("loaded language embeddings")

criterion1 = nn.MSELoss().to(device)

def get_text_data(target, labels_emb):
    return labels_emb[target].view(target.shape[0], text_emb_input_size).float()

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_models(load_epoch):
# load_epoch = 8499
    se_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/se_trip'+str(load_epoch)+'.pth.tar'
    sd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/sd_trip'+str(load_epoch)+'.pth.tar'
    te_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/te_trip'+str(load_epoch)+'.pth.tar'
    td_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/td_trip'+str(load_epoch)+'.pth.tar'

    sequence_encoder.load_state_dict(torch.load(se_checkpoint)['state_dict'])
    sequence_decoder.load_state_dict(torch.load(sd_checkpoint)['state_dict'])
    text_encoder.load_state_dict(torch.load(te_checkpoint)['state_dict'])
    text_decoder.load_state_dict(torch.load(td_checkpoint)['state_dict'])
    return 


def train_one_cycle(cycle_num, cycle_length):
    s_epoch = (cycle_num)*(cycle_length)
    e_epoch = (cycle_num+1)*(cycle_length)
    if cycle_length == 1700:
        cr_fact_epoch = 1400
    else:
        cr_fact_epoch = 1500
        
    for epoch in range(s_epoch, e_epoch):
        losses = AverageMeter()
        ce_loss_vals = []
        sequence_encoder.train()
        sequence_decoder.train()    
        text_encoder.train()
        text_decoder.train()
        k_trip = 0
        k_fact = max((0.1*(epoch - (s_epoch+1000))/3000), 0)
        k_fact2 = k_fact*(epoch > (s_epoch + cr_fact_epoch))
        cr_fact = 1*(epoch > (s_epoch + cr_fact_epoch))
        lw_fact = 0
        (inputs, target) = next(iter(train_loader))
        s = inputs.to(device)
        t = get_text_data(target, labels_emb).to(device)
        smu, slv = sequence_encoder(s)
        sz = reparameterize(smu, slv)
        sout = sequence_decoder(sz)

        tmu, tlv = text_encoder(t)
        tz = reparameterize(tmu, tlv)
        tout = text_decoder(tz)

        # cross reconstruction
        tfroms = text_decoder(sz)
        sfromt = sequence_decoder(tz)

        s_recons = criterion1(s, sout)
        t_recons = criterion1(t, tout)
        s_kld = KL_divergence(smu, slv).to(device) 
        t_kld = KL_divergence(tmu, tlv).to(device)
        s_crecons = criterion1(s, sfromt)
        t_crecons = criterion1(t, tfroms)
        l_wass = Wasserstein_distance(smu, slv, tmu, tlv)


        loss = s_recons + t_recons 
        loss -= k_fact*(s_kld)
        loss -= k_fact2*(t_kld)
        loss += cr_fact*(s_crecons + t_crecons)
        loss += lw_fact*(l_wass)

        # backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        ce_loss_vals.append(loss.cpu().detach().numpy())
        if epoch % 1 == 0:
            print('Epoch-{:<3d} \t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, loss=losses))
            print('srecons {:.4f}\ttrecons {:.4f}\t'.format(s_recons.item(), t_recons.item()))
            print('skld {:.4f}\ttkld {:.4f}\t'.format(s_kld.item(), t_kld.item()))
            print('screcons {:.4f}\ttcrecons {:.4f}\t'.format(s_crecons.item(), t_crecons.item()))        
            print('lwass {:.4f}\t'.format(l_wass.item()))

    return   
    
def save_model(epoch): 
    se_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/se_trip'+str(epoch)+'.pth.tar'
    sd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/sd_trip'+str(epoch)+'.pth.tar'
    te_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/te_trip'+str(epoch)+'.pth.tar'
    td_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/td_trip'+str(epoch)+'.pth.tar'
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': sequence_encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, se_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': sequence_decoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, sd_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': text_encoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, te_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': text_decoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, td_checkpoint)
    
    return



def train_classifier():

    cls = MLP([latent_size, ss]).to(device)
    print(load_classifier)
    if load_classifier == True:
        cls_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/clasifier.pth.tar'
        cls.load_state_dict(torch.load(cls_checkpoint)['state_dict'])
    else:
        cls_optimizer = optim.Adam(cls.parameters(), lr = 0.001)
        with torch.no_grad():
            c_t = unseen_labels_emb.to(device)
            c_t = c_t.repeat([500, 1])
            y = torch.tensor(range(ss)).to(device)
            y = y.repeat([500])
            text_encoder.eval()
            t_tmu, t_tlv = text_encoder(c_t)
            t_z = reparameterize(t_tmu, t_tlv)
            v_t = unseen_labels_emb.to(device).repeat([100, 1])
            v_y = torch.tensor(range(ss)).to(device).repeat([100])
            v_tmu, v_tlv = text_encoder(v_t)

        criterion2 = nn.CrossEntropyLoss().to(device)
        best = 0
        for c_e in range(300):
            cls.train()
            out = cls(t_z)
            c_loss = criterion2(out, y)
            cls_optimizer.zero_grad()
            c_loss.backward()
            cls_optimizer.step()
            c_acc = float(torch.sum(y == torch.argmax(out, -1)))/out.shape[0]
            if c_e % 50 == 0:
                print("Train Loss :", c_loss.item(), "Train Accuracy:", c_acc)

    cls.eval()

    u_inds = torch.from_numpy(unseen_inds)
    final_embs = []
    with torch.no_grad():
        sequence_encoder.eval()
        cls.eval()
        count = 0
        num = 0
        preds = []
        tars = []
        for (inp, target) in zsl_loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            final_embs.append(t_smu)
            t_out = cls(t_smu)
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)
        
    zsl_accuracy = float(count)/num
    final_embs = np.array([j.cpu().numpy() for i in final_embs for j in i])
    p = [j.item() for i in preds for j in i]
    t = [j.item() for i in tars for j in i]
    p = np.array(p)
    t = np.array(t)

    val_out_embs = []
    with torch.no_grad():
        sequence_encoder.eval()
        cls.eval()
        count = 0
        num = 0
        preds = []
        tars = []
        for (inp, target) in val_loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            t_out = cls(t_smu)
            val_out_embs.append(F.softmax(t_out))
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)
    #     print(float(count)/num)
    
    val_out_embs = np.array([j.cpu().numpy() for i in val_out_embs for j in i])
    return zsl_accuracy, val_out_embs, cls

def get_seen_zs_embeddings(cls):
    final_embs = []
    out_val_embeddings = []
    u_inds = torch.from_numpy(unseen_inds)
    with torch.no_grad():
        sequence_encoder.eval()
        cls.eval()
        count = 0
        num = 0
        preds = []
        tars = []
        for (inp, target) in val_loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            final_embs.append(t_smu)
            t_out = cls(t_smu)
            out_val_embeddings.append(F.softmax(t_out))
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)

    out_val_embeddings = np.array([j.cpu().numpy() for i in out_val_embeddings for j in i])
    return out_val_embeddings

def save_classifier(cls):
    cls_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/clasifier.pth.tar'
    save_checkpoint({
        'state_dict': cls.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, cls_checkpoint)
    
if __name__ == "__main__":
    
    best = 0
    if mode == 'eval':
        if load_epoch != None:
            load_models(load_epoch)
            zsl_acc, val_out_embs, _ = train_classifier()
            print('zsl accuracy ', zsl_acc)
        else:
            print('Mention Epoch to Load')
    else:
        if load_epoch != None:
            load_models(load_epoch)
        else:
            load_epoch = -1
        for num_cycle in range((load_epoch+1)//cycle_length, num_cycles):
#             train_one_cycle(num_cycle, cycle_length)
#             save_model(cycle_length*(num_cycle+1)-1)
            load_models(1699)
            zsl_acc, val_out_embs, cls = train_classifier()
            
                
            if (zsl_acc > best):
                best = zsl_acc
                save_classifier(cls)
                print('zsl_accuracy increased to ', best, ' on cycle ', num_cycle)
                print('saved checkpoint')
                if phase == 'train':
                    np.save('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/cadavae_' + str(ss) + '_r_gzsl_zs.npy', val_out_embs)
                else:
                    np.save('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/cadavae_' + str(ss) + '_r_unseen_zs.npy', val_out_embs)
                    seen_zs_embeddings = get_seen_zs_embeddings(cls)
                    np.save('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/cadavae_' + str(ss) + '_r_seen_zs.npy', seen_zs_embeddings)
