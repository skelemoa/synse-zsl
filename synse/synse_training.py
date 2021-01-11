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
from synse import Encoder, Decoder, KL_divergence, Wasserstein_distance, reparameterize, MLP

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
parser.add_argument('--load_classifier', type=bool, help="load classifier", default=False)
args = parser.parse_args()

# Arguments for 55-5 split
# gpu = '0'
# ss = 5
# st = 'r'
# dataset_path = 'ntu_results/shift_5_r'
# wdir = 'pos_aware_cada_vae_concatenated_latent_space_shift_5_r'
# le = 'bert'
# ve = 'shift'
# phase = 'train'
# num_class = 60

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
    

if 'bert' in le:
    text_emb_input_size = 1024
    # verb_emb_input_size = 1024
elif le == 'w2v':
    text_emb_input_size = 300
    # verb_emb_input_size = 300
else:
    pass

sequence_encoder = Encoder([vis_emb_input_size, latent_size]).to(device)
sequence_decoder = Decoder([latent_size, vis_emb_input_size]).to(device)
v_text_encoder = Encoder([text_emb_input_size, latent_size//2]).to(device)
v_text_decoder = Decoder([latent_size//2, text_emb_input_size]).to(device)

n_text_encoder = Encoder([text_emb_input_size, latent_size//2]).to(device)
n_text_decoder = Decoder([latent_size//2, text_emb_input_size]).to(device)

params = []
for model in [sequence_encoder, sequence_decoder, v_text_encoder, v_text_decoder, n_text_encoder, n_text_decoder]:
    params += list(model.parameters())

optimizer = optim.Adam(params, lr = 0.0001)

ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_train_loader(64, 8)
zsl_loader = ntu_loaders.get_val_loader(64, 8)
val_loader = ntu_loaders.get_test_loader(64, 8)
train_size = ntu_loaders.get_train_size()
zsl_size = ntu_loaders.get_val_size()
val_size = ntu_loaders.get_test_size()
print('Train on %d samples, validate on %d samples' % (train_size, val_size))


labels = np.load('../resources/labels.npy')
nouns_vocab = np.load('../resources/nouns_vocab.npy')
nouns_ohe = np.load('../resources/nouns_ohe.npy')
verbs_vocab = np.load('../resources/verbs_vocab.npy')
verbs_ohe = np.load('../resources/verbs_ohe.npy')
nouns = nouns_vocab[np.argmax(nouns_ohe, -1)]
verbs = verbs_vocab[np.argmax(verbs_ohe, -1)]

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

seen_verbs = verbs[seen_inds]
unseen_verbs = verbs[unseen_inds]

seen_nouns = nouns[seen_inds]
unseen_nouns = nouns[unseen_inds]

nouns_emb = torch.from_numpy(np.load('../resources/ntu' + str(num_class) + '_' + le + '_noun.npy')).view([num_class, text_emb_input_size])
nouns_emb = nouns_emb/torch.norm(nouns_emb, dim = 1).view([num_class, 1]).repeat([1, text_emb_input_size])

verbs_emb = torch.from_numpy(np.load('../resources/ntu' + str(num_class) + '_'  + le + '_verb.npy')).view([num_class, text_emb_input_size])
verbs_emb = verbs_emb/torch.norm(verbs_emb, dim = 1).view([num_class, 1]).repeat([1, text_emb_input_size])

unseen_nouns_emb = nouns_emb[unseen_inds, :]
seen_nouns_emb = nouns_emb[seen_inds, :]
unseen_verbs_emb = verbs_emb[unseen_inds, :]
seen_verbs_emb = verbs_emb[seen_inds, :]
print("loaded language embeddings")

criterion1 = nn.MSELoss().to(device)

def get_text_data(target):
    return nouns_emb[target].view(target.shape[0], text_emb_input_size).float(), verbs_emb[target].view(target.shape[0], text_emb_input_size).float()

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_models(load_epoch):
    se_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/se_'+str(load_epoch)+'.pth.tar'
    sd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/sd_'+str(load_epoch)+'.pth.tar'
    vte_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tve_'+str(load_epoch)+'.pth.tar'
    vtd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tvd_'+str(load_epoch)+'.pth.tar'
    nte_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tne_'+str(load_epoch)+'.pth.tar'
    ntd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tnd_'+str(load_epoch)+'.pth.tar'

    sequence_encoder.load_state_dict(torch.load(se_checkpoint)['state_dict'])
    sequence_decoder.load_state_dict(torch.load(sd_checkpoint)['state_dict'])
    v_text_encoder.load_state_dict(torch.load(vte_checkpoint)['state_dict'])
    v_text_decoder.load_state_dict(torch.load(vtd_checkpoint)['state_dict'])
    n_text_encoder.load_state_dict(torch.load(nte_checkpoint)['state_dict'])
    n_text_decoder.load_state_dict(torch.load(ntd_checkpoint)['state_dict'])
    
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

        # verb models
        sequence_encoder.train()
        sequence_decoder.train()    
        v_text_encoder.train()
        v_text_decoder.train()

        # hyper params
        k_fact = max((0.1*(epoch- (s_epoch+1000))/3000), 0)
        cr_fact = 1*(epoch > (s_epoch + cr_fact_epoch))
        v_k_fact2 = max((0.1*(epoch - (s_epoch + cr_fact_epoch))/3000), 0)*(cycle_num>1)
        n_k_fact2 = max((0.1*(epoch - ((s_epoch + cr_fact_epoch)))/3000), 0)*(cycle_num>1)
        v_cr_fact = 1*(epoch > (s_epoch + cr_fact_epoch))
        n_cr_fact = 1*(epoch > (s_epoch + cr_fact_epoch))
        
        # noun models
        n_text_encoder.train()
        n_text_decoder.train()


        (inputs, target) = next(iter(train_loader))
        s = inputs.to(device)
        nt, vt = get_text_data(target)
        nt = nt.to(device)
        vt = vt.to(device)

        smu, slv = sequence_encoder(s)
        sz = reparameterize(smu, slv)
        sout = sequence_decoder(sz)

        # noun forward pass

        ntmu, ntlv = n_text_encoder(nt)
        ntz = reparameterize(ntmu, ntlv)
        ntout = n_text_decoder(ntz)

        ntfroms = n_text_decoder(sz[:,:latent_size//2])

        s_recons = criterion1(s, sout)
        nt_recons = criterion1(nt, ntout)
        s_kld = KL_divergence(smu, slv).to(device) 
        nt_kld = KL_divergence(ntmu, ntlv).to(device)
        nt_crecons = criterion1(nt, ntfroms)
        

        # verb forward pass
        vtmu, vtlv = v_text_encoder(vt)
        vtz = reparameterize(vtmu, vtlv)
        vtout = v_text_decoder(vtz)

        vtfroms = v_text_decoder(sz[:,latent_size//2:])
        vt_recons = criterion1(vt, vtout)
        vt_kld = KL_divergence(vtmu, vtlv).to(device)
        vt_crecons = criterion1(vt, vtfroms)
        
        sfromt = sequence_decoder(torch.cat([ntz, vtz], 1))
        s_crecons = criterion1(s, sfromt)

        loss = s_recons + vt_recons + nt_recons 
        loss -= k_fact*(s_kld) + v_k_fact2*(vt_kld) + n_k_fact2*(nt_kld)
        loss += n_cr_fact*(nt_crecons) + v_cr_fact*(vt_crecons) + cr_fact*(s_crecons)
        
        # backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        ce_loss_vals.append(loss.cpu().detach().numpy())
        if epoch % 100 == 0:
            print('---------------------')
            print('Epoch-{:<3d} \t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, loss=losses))
            print('srecons {:.4f}\t ntrecons {:.4f}\t vtrecons {:.4f}\t'.format(s_recons.item(), nt_recons.item(), vt_recons.item()))
            print('skld {:.4f}\t ntkld {:.4f}\t vtkld {:.4f}\t'.format(s_kld.item(), nt_kld.item(), vt_kld.item()))
            print('screcons {:.4f}\t ntcrecons {:.4f}\t ntcrecons {:.4f}\t'.format(s_crecons.item(), nt_crecons.item(), vt_crecons.item()))        
#             print('nlwass {:.4f}\t vlwass {:.4f}\n'.format(nl_wass.item(), vl_wass.item()))

    return 

def save_model(epoch):
    se_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/se_'+str(epoch)+'.pth.tar'
    sd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/sd_'+str(epoch)+'.pth.tar'
    tve_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tve_'+str(epoch)+'.pth.tar'
    tvd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tvd_'+str(epoch)+'.pth.tar'
    tne_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tne_'+str(epoch)+'.pth.tar'
    tnd_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/tnd_'+str(epoch)+'.pth.tar'

    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': sequence_encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, se_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': sequence_decoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, sd_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': v_text_encoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, tve_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': v_text_decoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, tvd_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': n_text_encoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, tne_checkpoint)
    save_checkpoint({ 'epoch': epoch + 1,
        'state_dict': n_text_decoder.state_dict(),
    #     'optimizer': optimizer.state_dict()
    }, tnd_checkpoint)



def train_classifier():

    cls = MLP([latent_size, ss]).to(device)
    if load_classifier == True:
        cls_checkpoint = '/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/clasifier.pth.tar'
        cls.load_state_dict(torch.load(cls_checkpoint)['state_dict'])
    else:
        cls_optimizer = optim.Adam(cls.parameters(), lr = 0.001)
        print('classifier_training ....')
        with torch.no_grad():
            n_t = unseen_nouns_emb.to(device).float()
            n_t = n_t.repeat([500, 1])
            v_t = unseen_verbs_emb.to(device).float()
            v_t = v_t.repeat([500, 1])
            y = torch.tensor(range(ss)).to(device)
            y = y.repeat([500])
            v_text_encoder.eval()
            n_text_encoder.eval() 
            nt_tmu, nt_tlv = n_text_encoder(n_t)
            vt_tmu, vt_tlv = v_text_encoder(v_t)
            vt_z = reparameterize(vt_tmu, vt_tlv)
            nt_z = reparameterize(nt_tmu, nt_tlv)

        criterion2 = nn.CrossEntropyLoss().to(device)
        best = 0

        for c_e in range(300):
            cls.train()
            out = cls(torch.cat([nt_z, vt_z], 1))
            c_loss = criterion2(out, y)
            cls_optimizer.zero_grad()
            c_loss.backward()
            cls_optimizer.step()
            c_acc = float(torch.sum(y == torch.argmax(out, -1)))/(ss*500)
            if c_e % 50 == 0:
                print("Train Loss :", c_loss.item(), "Train Accuracy:", c_acc)

    cls.eval()

#     print(unseen_inds)
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
            nt_smu, t_slv = sequence_encoder(t_s)
            final_embs.append(nt_smu)
            t_out = cls(nt_smu)
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)
#         print(float(count)/num)

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
        gzsl_count = 0
        gzsl_num = 0
        gzsl_preds = []
        gzsl_tars = []
        for (inp, target) in val_loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            t_out = cls(t_smu)
            val_out_embs.append(F.softmax(t_out, 1))
            pred = torch.argmax(t_out, -1)
            gzsl_preds.append(u_inds[pred])
            gzsl_tars.append(target)
            gzsl_count += torch.sum(u_inds[pred] == target)
            num += len(target)
#         print(float(count)/num)

    
    val_out_embs = np.array([j.cpu().numpy() for i in val_out_embs for j in i])
#     np.save('/ssd_scratch/cvit/pranay.gupta/unseen_out/synse_5_r_gzsl_zs.npy', val_out_embs)
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
    #         t_sz = reparameterize(t_smu, t_slv)
            final_embs.append(t_smu)
            t_out = cls(t_smu)
            out_val_embeddings.append(F.softmax(t_out))
            pred = torch.argmax(t_out, -1)
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)
#         print(float(count)/num)

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
            train_one_cycle(num_cycle, cycle_length)
            save_model(cycle_length*(num_cycle+1)-1)
            zsl_acc, val_out_embs, cls = train_classifier()
            
                
            if (zsl_acc > best):
                best = zsl_acc
                save_classifier(cls)
                print('zsl_accuracy increased to ', best, ' on cycle ', num_cycle)
                print('saved checkpoint')
                if phase == 'train':
                    np.save('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/synse_' + str(ss) + '_r_gzsl_zs.npy', val_out_embs)
                else:
                    np.save('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/synse_' + str(ss) + '_r_unseen_zs.npy', val_out_embs)
                    seen_zs_embeddings = get_seen_zs_embeddings(cls)
                    np.save('/ssd_scratch/cvit/pranay.gupta/language_modelling/' + wdir + '/' + le + '/synse_' + str(ss) + '_r_seen_zs.npy', seen_zs_embeddings)
                    
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
