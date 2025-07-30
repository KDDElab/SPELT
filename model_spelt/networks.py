import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
plt.rcParams['font.serif'] = ['Times New Roman']


class SPELTnet(nn.Module):
    def __init__(self, n_feature, n_linear, n_hyperedge=20):
        super(SPELTnet, self).__init__()
        self.hyper_proj = HypergraphProjection(n_feature, n_linear, n_hyperedge)
        self.embedding_history = []

    def forward(self, anchor, positive, negative):
        # Hypergraph mapping
        anchor_emb = self.hyper_proj(anchor)
        positive_emb = self.hyper_proj(positive)
        negative_emb = self.hyper_proj(negative)

        #  Storage embedded for visualisation
        self.embedding_history.append({
            'anchor': anchor_emb.detach().cpu().numpy(),
            'positive': positive_emb.detach().cpu().numpy(),
            'negative': negative_emb.detach().cpu().numpy()
        })

        
        dis = F.pairwise_distance(anchor_emb, negative_emb) - F.pairwise_distance(anchor_emb, positive_emb)
        
        return anchor_emb, positive_emb, negative_emb, dis
    # =============== Visualisation methods ===============
    def plot_embedding_space(self):
        if not self.embedding_history:
            print("No embedding data available")
            return
        
        all_embeddings = []
        labels = []
        
        for batch in self.embedding_history:
            n_samples = batch['anchor'].shape[0]
            
            all_embeddings.append(batch['anchor'])
            labels.extend(['Anchor'] * n_samples)
            
            all_embeddings.append(batch['positive'])
            labels.extend(['Positive'] * n_samples)
            
            all_embeddings.append(batch['negative'])
            labels.extend(['Negative'] * n_samples)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        
        color_map = {'Anchor': '#E74C3C', 'Positive': '#2ECC71', 'Negative': '#3498DB'}
        colors = [color_map[label] for label in labels]
        
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=colors, alpha=0.7, s=80, edgecolor='w', linewidth=0.5)
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                  markersize=10, label=label) for label, color in color_map.items()]
        plt.legend(handles=handles, loc='best')
        
        plt.title("t-SNE Projection of Embedding Space", fontsize=16)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    def visualize_all(self):
        self.hyper_proj.plot_attention_heatmap()
        self.hyper_proj.plot_feature_interaction()
        self.hyper_proj.plot_hypergraph_structure()
        self.plot_embedding_space()

    def get_lnr(self, x):
        return self.hyper_proj(x)
    

class HypergraphConv(nn.Module):
    """Standard hypergraph convolution layer"""
    def __init__(self, in_features, out_features, use_attention=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.attn_weight = nn.Parameter(torch.Tensor(out_features, 1)) if use_attention else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.attn_weight is not None:
            nn.init.xavier_uniform_(self.attn_weight)

    def forward(self, X, H):
        """
        X: Node features (n_nodes, in_features)
        H: (n_nodes, n_hyperedges)
        """
        # Hypergraph normalisation
        D_v = torch.diag(1.0 / (torch.sum(H, dim=1) + 1e-6))  
        D_e = torch.diag(1.0 / (torch.sum(H, dim=0) + 1e-6))  
        
        # Hypergraph message propagation
        X = torch.matmul(D_v, torch.matmul(H, torch.matmul(D_e, torch.matmul(H.t(), X))))
        
        # Feature transformation
        X = torch.matmul(X, self.weight)
        return X
    
class HypergraphProjection(nn.Module):
    def __init__(self, in_features, out_features, n_hyperedge=20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features, dtype=torch.float32),  
            nn.Tanh()
        )
        
        self.hyperedge_centers = nn.Parameter(torch.randn(n_hyperedge, out_features, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.K = 5  
        self.out_features = out_features

        self.feature_names = [f'F{i}' for i in range(in_features)]
        self.hyperedge_names = [f'HE{i}' for i in range(n_hyperedge)]

        self.attention_history = []
        self.hypergraph_history = []

    def _build_sparse_hypergraph(self, x):
        """More efficient construction of sparse hypergraphs"""
        x = x.to(torch.float32) if x.dtype != torch.float32 else x
        
        n_nodes = x.size(0)
        device = x.device

        def safe_normalize(tensor):
            norm = torch.norm(tensor, p=2, dim=1, keepdim=True)
            norm = torch.where(norm > 0, norm, torch.ones_like(norm))
            return tensor / norm
        
        norm_centers = safe_normalize(self.hyperedge_centers)
        norm_x = safe_normalize(x)
        similarities = torch.mm(norm_x, norm_centers.t())  # [n_nodes, n_hyperedge]
        similarities = torch.clamp(similarities, min=-1.0, max=1.0)  
        
        topk_values, topk_indices = torch.topk(similarities, k=self.K, dim=1)

        attention_matrix = torch.zeros(n_nodes, self.hyperedge_centers.size(0), 
                                      device=device, dtype=torch.float32)
        attention_matrix[torch.arange(n_nodes).repeat_interleave(self.K), 
                        topk_indices.view(-1)] = topk_values.view(-1)
        self.attention_history.append(attention_matrix.detach().cpu().numpy())
        
        row_indices = torch.arange(n_nodes, device=device).repeat_interleave(self.K)
        col_indices = topk_indices.view(-1)
        values = topk_values.view(-1)
        
        return torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]), 
            values,
            (n_nodes, self.hyperedge_centers.size(0)),
            dtype=torch.float32 
        )
    
    def _equivariant_diffusion(self, X, H_sparse):
        """Memory-friendly diffusion implementation"""
        X = X.to(torch.float32) if X.dtype != torch.float32 else X
        
        n_nodes = X.size(0)
        
        D_v = torch.sparse.sum(H_sparse, dim=1).to_dense()
        D_v_inv_sqrt = 1.0 / (torch.sqrt(D_v) + 1e-6)
        
        D_e = torch.sparse.sum(H_sparse, dim=0).to_dense()
        D_e_inv = 1.0 / (D_e + 1e-6)
        
        # step 2: H^T * (D_v^{-1/2} * X)
        step1 = X * D_v_inv_sqrt.unsqueeze(1)
        step2 = torch.sparse.mm(H_sparse.t(), step1)
        
        # step 3: D_e^{-1} * step2
        step3 = step2 * D_e_inv.unsqueeze(1)
        
        # step 4: H * step3
        step4 = torch.sparse.mm(H_sparse, step3)
        
        # step 5: D_v^{-1/2} * step4
        step5 = step4 * D_v_inv_sqrt.unsqueeze(1)

        if len(self.hypergraph_history) < 5:  
            hypergraph_data = {
                'H_sparse': H_sparse.detach().cpu(),
                'D_v': D_v.detach().cpu(),
                'D_e': D_e.detach().cpu(),
                'X': X.detach().cpu()
            }
            self.hypergraph_history.append(hypergraph_data)
        
        return (1 - self.alpha) * X + self.alpha * step5

    def forward(self, x):
        x = x.to(torch.float32) if x.dtype != torch.float32 else x
        
        x_proj = self.mlp(x)
        
        H_sparse = self._build_sparse_hypergraph(x_proj)
        
        
        x_hyper = self._equivariant_diffusion(x_proj, H_sparse)
        
        return (x_proj + x_hyper).to(x.dtype)
    
    def compute_total_weight(self):
        """ (x_proj + x_hyper)"""
        
        W1 = self.mlp[0].weight.data  # [out_features, in_features]
        
        I = torch.eye(self.out_features, device=W1.device, dtype=torch.float32)
        
        W_diff = (2 - self.alpha) * I + self.alpha * I
        
        W_total = torch.mm(W_diff, W1)
        
        return W_total
    
    def plot_attention_heatmap(self):
        if not self.attention_history:
            print("No attention data available")
            return
        
        weights = self.attention_history[-1]
        avg_weights = np.mean(weights, axis=0)
        sorted_idx = np.argsort(avg_weights)[::-1]
        
        plt.figure(figsize=(12, 8))
        colors = ["#2E86C1", "#85C1E9", "#F9E79F", "#F39C12", "#E74C3C"]
        cmap = LinearSegmentedColormap.from_list("attention_cmap", colors)
        
        sns.heatmap(weights[:, sorted_idx], cmap=cmap, 
                    yticklabels=False, xticklabels=np.array(self.hyperedge_names)[sorted_idx])
        plt.title("Feature-Hyperedge Attention Weights", fontsize=14)
        plt.xlabel("Hyperedges (Sorted by Importance)", fontsize=12)
        plt.ylabel("Samples", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('Feature-Hyperedge-Attention-Weights.png',format='png',dpi=1024)
        plt.show()
    
    def plot_feature_interaction(self):
        if not self.attention_history:
            print("No attention data available")
            return
        
        weights = np.concatenate(self.attention_history, axis=0)
        corr_matrix = np.corrcoef(weights.T)
        np.fill_diagonal(corr_matrix, 0) 
        
        plt.figure(figsize=(14, 10))
        G = nx.Graph()
        edge_weights = []
        
    
        for i, name in enumerate(self.feature_names):
            G.add_node(name, size=1000 * np.mean(weights[:, i]))
        
        
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                if corr_matrix[i, j] > 0.2:  
                    G.add_edge(self.feature_names[i], self.feature_names[j], 
                              weight=5 * abs(corr_matrix[i, j]))
                    edge_weights.append(abs(corr_matrix[i, j]))
        
    
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
        edge_widths = [G.edges[e]['weight'] for e in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='#3498DB', alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              edge_color='#7D3C98', alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title("High-Order Feature Interactions", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('High-Order-Feature-Interactions.png',format='png',dpi=1024)
        plt.show()
    
    def plot_hypergraph_structure(self):
        if not self.hypergraph_history:
            print("No hypergraph data available")
            return
        
        hypergraph_data = self.hypergraph_history[0]
        H_sparse = hypergraph_data['H_sparse']
        D_v = hypergraph_data['D_v']
        D_e = hypergraph_data['D_e']
        
        H_dense = H_sparse.to_dense().numpy()
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.weight'] = 'bold'
        
        plt.figure(figsize=(14, 10))
        
        
        plt.subplot(221)
        sns.histplot(D_v.numpy(), bins=20, kde=True, color='#3498DB')
        plt.title('Node Degree Distribution', fontsize=22)
        plt.xlabel('Degree', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        
        
        plt.subplot(222)
        sns.histplot(D_e.numpy(), bins=20, kde=True, color='#E74C3C')
        plt.title('Hyperedge Degree Distribution', fontsize=22)
        plt.xlabel('Degree', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        
        
        plt.subplot(223)
        sns.heatmap(H_dense, cmap='viridis', cbar_kws={'label': 'Association Strength'})
        plt.title('Hypergraph Association Matrix', fontsize=22)
        plt.xlabel('Hyperedges', fontsize=20)
        plt.ylabel('Nodes', fontsize=20)
        
        
        plt.subplot(224)
        hyperedge_importance = H_dense.sum(axis=0)
        plt.bar(range(len(hyperedge_importance)), hyperedge_importance, color='#2ECC71')
        plt.title('Hyperedge Importance', fontsize=22)
        plt.xlabel('Hyperedge Index', fontsize=20)
        plt.ylabel('Total Association', fontsize=20)
        
        plt.tight_layout()
        plt.suptitle('Hypergraph Structure Analysis', fontsize=16, y=1.02)
        plt.savefig('Hypergraph-Structure-Analysis-mnist.pdf',format='pdf',dpi=1024)
        plt.show()

    
class MyLoss_imp(nn.Module):
    """
    Triplet-constrained Density-aware Loss
    """
    def __init__(self, alpha1=0.8, alpha2=0.2, margin=2.0, lambda_reg=0.1):
        super().__init__()
        self.alpha1 = alpha1    
        self.alpha2 = alpha2    
        self.lambda_reg = lambda_reg  
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin, p=2)
        
        
        self.typicality_beta = nn.Parameter(torch.tensor(1.0))
        self.typicality_gamma = nn.Parameter(torch.tensor(0.5))
    
    def _compute_typicality(self, embeddings):
        pairwise_dist = torch.cdist(embeddings, embeddings)
        kernel = torch.exp(-self.typicality_beta * pairwise_dist.pow(2))
        density = torch.mean(kernel, dim=1)
        return torch.sigmoid(self.typicality_gamma * density)
    
    
    def _compute_typicality(self, embeddings):
        
        pairwise_dist = torch.cdist(embeddings, embeddings)
        pairwise_dist = torch.clamp(pairwise_dist, min=1e-6)
        
        
        max_val = torch.max(pairwise_dist).item()
        exp_safe = torch.exp(-self.typicality_beta * pairwise_dist.pow(2) / max(1.0, max_val))
        
        density = torch.mean(exp_safe, dim=1)
        return torch.sigmoid(self.typicality_gamma * density)

    def forward(self, embed_anchor, embed_pos, embed_neg, 
                inlier_embeddings=None):
        
        loss_tml = self.criterion_tml(embed_anchor, embed_pos, embed_neg)
        
        typicality_neg = self._compute_typicality(embed_neg)
        typicality_anchor = self._compute_typicality(embed_anchor)
        typicality_pos = self._compute_typicality(embed_pos)
        typ_diff = torch.abs(typicality_neg - typicality_anchor)
        

        if inlier_embeddings is not None:
            inlier_typicality = self._compute_typicality(embed_pos)
            reg_loss = torch.mean(1.0 - inlier_typicality) # + torch.mean(typicality_anchor)
        else:
            reg_loss = 0.0

        total_loss = (
            self.alpha1 * loss_tml +
            self.alpha2 * torch.mean(typ_diff) +
            self.lambda_reg * reg_loss
        )
        return total_loss, typ_diff.mean()