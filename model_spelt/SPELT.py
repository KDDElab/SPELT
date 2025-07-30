import torch
print(torch.__version__)          
print(torch.cuda.is_available())  
print(torch.cuda.device_count())  

from pyod.models import lscp
import numpy as np
import time, math
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint

from torch.optim import lr_scheduler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from tqdm import tqdm
from model_spelt.utils import EarlyStopping, min_max_normalize

from model_spelt.datasets import MyHardSingleTripletSelector, MyHardSingleTripletSelector1, EDHNNTripletSelector
from model_spelt.datasets import SingleTripletDataset
from model_spelt.networks import SPELTnet
from model_spelt.networks import MyLoss_imp


class SPELT:
    def __init__(self, nbrs_num=30, rand_num=30, alpha1=0.8, alpha2=0.2,
                 n_epoch=10, batch_size=64, lr=0.1, n_linear=64, margin=2.,
                 verbose=True, gpu=True):
        self.verbose = verbose
        self.current_model = None

        self.x = None
        self.y = None
        self.ano_idx = None
        self.dim = None

        # a list of normal nbr of each anomaly
        self.normal_nbr_indices = []

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda and gpu else "cpu")
        if cuda:
            torch.cuda.set_device(1)
        print("device:", self.device)

        self.nbrs_num = nbrs_num
        self.rand_num = rand_num
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_linear = n_linear
        self.margin = margin
        self.feature_weights_history = []
        return

    def fit(self, x, y):
        device = self.device

        self.dim = x.shape[1]
        x = min_max_normalize(x)
        self.ano_idx = np.where(y == 1)[0]

        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)
        self.prepare_nbrs()

        # train model for each anomaly
        W_lst = []
        if self.verbose:
            iterator = range(len(self.ano_idx))
        else:
            iterator = tqdm(range(len(self.ano_idx)))
        for ii in iterator:
            idx = self.ano_idx[ii]

            s_t = time.time()
            W, losses = self.interpret_ano(ii)
        
            if np.isnan(W).any():
                print(f"Warning: NaN detected in weights for sample {ii}")
                W = np.ones_like(W) 
        

            W_lst.append(W)
            '''
            print("losses", losses)
            plt.plot(range(len(losses)), losses, label="Loss")
            plt.xlabel("Epoch", fontsize = 16)
            plt.ylabel("Loss", fontsize = 16)
            plt.xticks(fontsize = 16)
            plt.yticks(fontsize = 16)               
            #plt.title("Loss Curve")
            plt.savefig('losses_imp.png',format='png',dpi=1024)#输出
            # plt.savefig('losses_imp.eps',format='eps',dpi=1024)
            plt.show()
            '''
            
            print("losses", losses)
            plt.figure(figsize=(10, 6), dpi=100) 

            
            # plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman']
            plt.rcParams['font.weight'] = 'bold'

            plt.plot(range(len(losses)), losses, 
                    color='#1f77b4',     #
                    linewidth=6.0,       # 
                    alpha=0.9,           #
                    marker='o',          #
                    markersize=6, 
                    markeredgecolor='darkblue',
                    markerfacecolor='lightblue',
                    label='Training Loss')

            
            plt.xlabel("Epoch", fontsize=30, fontweight='bold')
            plt.ylabel("Loss", fontsize=30, fontweight='bold')

            
            plt.xticks(fontsize=26, fontweight='bold')
            plt.yticks(fontsize=26, fontweight='bold')

        
            plt.grid(True, linestyle='--', alpha=0.7)

            
            plt.legend(fontsize=30, loc='upper right')

            
            plt.xlim(0, len(losses)-1)
            if losses:  
                y_min = min(losses) * 0.95
                y_max = max(losses) * 1.05
                plt.ylim(y_min, y_max)

            
            plt.tight_layout()
            plt.savefig('fault.png', format='png', dpi=1024, bbox_inches='tight')
            plt.savefig('fault.pdf', format='pdf', dpi=1200)  
            plt.show()
            
            

            if self.verbose:
                print("Ano_id:[{}], ({}/{}) \t time: {:.2f}s\n".format(
                    idx, (ii + 1), len(self.ano_idx),
                    (time.time() - s_t)))

        fea_weight_lst = []

        for ii, idx in enumerate(W_lst):
            w = W_lst[ii]
            fea_weight = np.zeros(self.dim)

            # attention (linear space) + w --> feature weight (original space)
            for j in range(len(w)):
                fea_weight += abs(w[j])
            fea_weight_lst.append(fea_weight)
        return fea_weight_lst
    
    def _get_inlier_embeddings(self):
        if self.current_model is None:
            raise ValueError("Model not initialized")
            
        noml_idx = np.where(self.y.cpu() == 0)[0]
        sample_idx = np.random.choice(
            noml_idx, 
            size=min(100, len(noml_idx)),
            replace=False
        )
        
        with torch.no_grad():
            return self.current_model.hyper_proj(
                self.x[sample_idx].to(self.device))

    def interpret_ano(self, ii):
        idx = self.ano_idx[ii]
        device = self.device
        dim = self.dim

        nbr_indices = self.normal_nbr_indices[ii]
        data_loader, test_loader = self.prepare_triplets(idx, nbr_indices)
        n_linear = self.n_linear
        model = SPELTnet(n_feature=dim, n_linear=n_linear)
        model.to(device)
        self.current_model = model

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion = MyLoss_imp(alpha1=self.alpha1, alpha2=self.alpha2, margin=self.margin)

        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
        early_stp = EarlyStopping(patience=3, verbose=False)


        losses = []
        for epoch in range(self.n_epoch):
            model.train()
            total_loss = 0
            total_typ_diff = 0
            es_time = time.time()

            batch_cnt = 0
            for anchor, pos, neg in data_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                embed_anchor, embed_pos, embed_neg, dis = model(anchor, pos, neg)
                # loss = criterion(embed_anchor, embed_pos, embed_neg, dis)
                inlier_embeds = self._get_inlier_embeddings()
                loss, typ_diff = criterion(embed_anchor, embed_pos, embed_neg,inlier_embeddings=inlier_embeds) 

                total_loss += loss
                total_typ_diff += typ_diff

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_cnt += 1

            train_loss = total_loss / batch_cnt
            est = time.time() - es_time

            if self.verbose and (epoch + 1) % 1 == 0:
                message = 'Epoch: [{:02}/{:02}]  loss: {:.4f} Time: {:.2f}s'.format(epoch + 1, self.n_epoch,
                                                                                    train_loss, est)
                print(message)
            scheduler.step()

            early_stp(train_loss, model)
            if early_stp.early_stop:
                model.load_state_dict(torch.load(early_stp.path))
                if self.verbose:
                    print("early stopping")
                break
            losses.append(total_loss.item())

            torch.cuda.empty_cache()

        for anchor, pos, neg in test_loader:
            model.eval()
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            _, _, _, _ = model(anchor, pos, neg)
        W_total = model.hyper_proj.compute_total_weight()
        W = np.abs(W_total.cpu().detach().numpy())

    
        self.feature_weights_history.append(W)
        
        
        if ii == 0:  
            model.visualize_all()
            
            self.plot_feature_weights_evolution()

        model.cpu()
        del model
        torch.cuda.empty_cache()

        return W, losses
    
    def plot_feature_weights_evolution(self):
        if len(self.feature_weights_history) < 2:
            print("Not enough epochs for evolution plot")
            return
        
        weights = np.array(self.feature_weights_history)
        n_features = weights.shape[2]
        
        plt.figure(figsize=(14, 8))
        
        for i in range(n_features):
            plt.plot(weights[:, 0, i], label=f'F{i}', alpha=0.8)
        
        plt.title('Feature Weight Evolution During Training', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Feature Weight', fontsize=12)
        plt.legend(title='Features', loc='upper right', ncol=4)
        plt.grid(alpha=0.3)
        plt.xlim(0, len(weights)-1)
        plt.tight_layout()
        plt.show()

    def prepare_triplets(self, idx, nbr_indices):
        x = self.x
        y = self.y
        # selector = MyHardSingleTripletSelector1(nbrs_num=self.nbrs_num, rand_num=self.rand_num, nbr_indices=nbr_indices, density_bandwidth=0.5, topk_ratio=0.6)
        
        selector = EDHNNTripletSelector(
            nbrs_num=self.nbrs_num, 
            rand_num=self.rand_num,
            nbr_indices=nbr_indices,
            diffusion_steps=2
        )
        
        
        dataset = SingleTripletDataset(idx, x, y, triplets_selector=selector)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset, batch_size=len(dataset))
        return data_loader, test_loader

    def prepare_nbrs(self):
        x = self.x.cpu().data.numpy()
        y = self.y.cpu().data.numpy()

        anom_idx = np.where(y == 1)[0]
        x_anom = x[anom_idx]
        noml_idx = np.where(y == 0)[0]
        x_noml = x[noml_idx]
        n_neighbors = self.nbrs_num

        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors).fit(x_noml)
        tmp_indices = nbrs_local.kneighbors(x_anom)[1]

        for idx in tmp_indices:
            nbr_indices = noml_idx[idx]
            self.normal_nbr_indices.append(nbr_indices)
        return