import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity

try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors


class SingleTripletDataset(Dataset):
    def __init__(self, anom_idx, x, y, triplets_selector, transform=None):
        self.transform = transform
        self.data = x
        self.triplets = triplets_selector.get_triplets(anom_idx, x, y)

    def __getitem__(self, index):
        a_idx, p_idx, n_idx = self.triplets[index]
        anchor, positive, negative = self.data[a_idx], self.data[p_idx], self.data[n_idx]
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative

    def __len__(self):
        return len(self.triplets)


class SingleDataset(Dataset):
    def __init__(self, anom_idx, x, y, data_selector, transform=None):
        self.transform = transform
        self.selected_data = data_selector.get_data(anom_idx, x, y)

    def __getitem__(self, index):
        data = self.selected_data[0][index]
        target = self.selected_data[1][index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.selected_data[0])


class MyHardSingleTripletSelector:
    def __init__(self, nbrs_num, rand_num, nbr_indices):
        self.x = None
        self.y = None
        self.nbrs_num = nbrs_num
        self.rand_num = rand_num
        self.nbr_indices = nbr_indices

    def get_triplets(self, anom_idx, x, y, normal_label=0):
        self.x = x.cpu().data.numpy()
        self.y = y.cpu().data.numpy()

        noml_idx = np.where(self.y == normal_label)[0]
        nbr_indices = self.nbr_indices
        rand_num = self.rand_num

        rand_canddt = np.setdiff1d(noml_idx, nbr_indices)
        rand_indices = np.random.choice(rand_canddt, rand_num, replace=False)

        triplets = [[anchor, positive, anom_idx]
                    for anchor in rand_indices
                    for positive in nbr_indices]
        return torch.LongTensor(np.array(triplets))

class DensityAwareTripletSelector:
    def __init__(self, bandwidth=0.2, n_samples=500):
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        
    def _estimate_density(self, data):
        """核密度估计"""
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(data)
        log_dens = kde.score_samples(data)
        return np.exp(log_dens)

class MyHardSingleTripletSelector1(MyHardSingleTripletSelector):
    def __init__(self, nbrs_num=30, rand_num=30, nbr_indices=None, 
                 density_bandwidth=0.5, topk_ratio=0.7):
        super().__init__(nbrs_num, rand_num, nbr_indices)
        
        # 密度估计参数
        self.density_estimator = KernelDensity(bandwidth=density_bandwidth)
        self.topk_ratio = topk_ratio

    def _get_high_density_samples(self, data, indices):
        if len(indices) == 0:
            return indices
            
        
        if USE_FAISS and data.shape[1] > 1:
            d = data.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(data[indices].astype('float32'))
            _, I = index.search(data[indices].astype('float32'), min(50, len(indices)))
            density_scores = np.array([len(set(I[i])) for i in range(len(indices))])
        else:
            
            self.density_estimator.fit(data[indices])
            log_dens = self.density_estimator.score_samples(data[indices])
            density_scores = np.exp(log_dens)
        
        n_select = max(1, int(len(indices) * self.topk_ratio))
        return indices[np.argpartition(-density_scores, n_select)[:n_select]]


    
    
    def _get_low_density_samples(self, data, indices):
        self.density_estimator.fit(data[indices])
        log_dens = self.density_estimator.score_samples(data[indices])
        density_scores = np.exp(log_dens)
        sorted_indices = np.argsort(density_scores)
        n_select = max(1, int(len(indices) * (1 - self.topk_ratio)))
        return indices[sorted_indices[:n_select]]
        
    def get_triplets(self, anom_idx, x, y, normal_label=0):
        self.x = x.cpu().data.numpy()
        self.y = y.cpu().data.numpy()

        noml_idx = np.where(self.y == normal_label)[0]
        nbr_indices = self.nbr_indices
        rand_num = self.rand_num

        
        high_density_nbrs = self._get_high_density_samples(self.x, noml_idx)
        
        
        rand_canddt = np.setdiff1d(noml_idx, high_density_nbrs)
        candidate_dens = self._get_high_density_samples(self.x, rand_canddt)
        low_density_indices = self._get_low_density_samples(self.x, candidate_dens)
        rand_indices = np.random.choice(
            low_density_indices, 
            min(rand_num, len(low_density_indices)), 
            replace=False
        )

    
        triplets = [[anchor, positive, anom_idx]
                   for anchor in rand_indices
                   for positive in high_density_nbrs]
                   
        return torch.LongTensor(np.array(triplets))

class EDHNNTripletSelector(MyHardSingleTripletSelector1):
    def __init__(self, nbrs_num=30, rand_num=30, nbr_indices=None, 
                 diffusion_steps=3):
        super().__init__(nbrs_num, rand_num, nbr_indices)
        self.diffusion_steps = diffusion_steps
    
    def _diffusion_based_sampling(self, data, indices):
        similarities = torch.cdist(data[indices], data[indices])
        topk_values, topk_indices = torch.topk(similarities, k=5, dim=1, largest=False)
        
        
        n = len(indices)
        indices_matrix = torch.stack([
            torch.arange(n).repeat_interleave(5),
            topk_indices.view(-1)
        ])
        values = torch.exp(-topk_values.view(-1))
        A = torch.sparse_coo_tensor(indices_matrix, values, (n, n))
        
        
        X = torch.eye(n).to(A.device)
        for _ in range(self.diffusion_steps):
            X = torch.sparse.mm(A, X)
        
        importance = torch.sum(X, dim=1)
        return indices[torch.topk(importance, k=self.nbrs_num//2, dim=0)[1]]