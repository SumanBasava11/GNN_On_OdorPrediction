import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import dropout_adj, to_dense_adj, dense_to_sparse
from torch_geometric.nn.pool import graclus
from torch_geometric.nn.conv import GCNConv
from torch_scatter import scatter_add

class DropPoolChannel(nn.Module):
    """
    Dropping Pooling channel:
    Drop nodes based on importance scores.
    Importance scores can be computed from topology or node features.
    """
    def __init__(self, node_dim, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(node_dim, 1)

    def forward(self, x, adj, mask=None):
        # x: [B, N, F]
        B, N, F = x.size()

        # Compute node scores (importance)
        scores = self.score_layer(x).squeeze(-1)  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))  # mask padding nodes

        # For each graph in batch, select top-k nodes by score
        k = int(self.ratio * N)
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)

        # Gather nodes and adjacency corresponding to topk nodes
        batch_indices = torch.arange(B, device=x.device).view(-1, 1).repeat(1, k)
        x_pooled = x[batch_indices, topk_indices]  # [B, k, F]

        # Extract adjacency matrix for pooled nodes
        adj_pooled = torch.zeros(B, k, k, device=x.device)
        for b in range(B):
            indices = topk_indices[b]
            adj_pooled[b] = adj[b][indices][:, indices]

        # New mask is all True (no padding in pooled graph)
        new_mask = torch.ones(B, k, dtype=torch.bool, device=x.device)

        return x_pooled, adj_pooled, new_mask
    
class CoarseningPoolChannel(nn.Module):
    """
    Coarsening pooling channel using Graclus clustering.
    Pools the graph by clustering nodes.
    """
    def __init__(self):
        super().__init__()
      
        self.refine_conv = GCNConv(in_channels=1, out_channels=1)

    def forward(self, x, adj, mask=None):
        # x: [B, N, F], adj: [B, N, N]

        B, N, F = x.size()
        x_list, adj_list = [], []
        mask_list = []

        for b in range(B):
            # Convert dense adj to edge_index
            edge_index, edge_weight = dense_to_sparse(adj[b])

            # Move to device
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)

            # Cluster assignment using graclus (returns cluster indices per node)
            cluster = graclus(edge_index, edge_weight, num_nodes=N).to(x.device)

            # Pool nodes by cluster: mean pooling
            num_clusters = cluster.max().item() + 1
            cluster = cluster.to(x.device)

            # Aggregate node features by cluster using scatter_add
            pooled_x = scatter_add(x[b], cluster.unsqueeze(-1).expand(-1, F), dim=0, dim_size=num_clusters)
            # Count nodes per cluster
            # x_b = x[b]
            count = scatter_add(torch.ones_like(cluster, dtype=torch.float, device=x.device), cluster, dim=0, dim_size=num_clusters)
            count = count.clamp(min=1).unsqueeze(-1)
            pooled_x = pooled_x / count  # mean pooling
            
            # Build pooled adjacency matrix:
            src_clusters = cluster[edge_index[0]]
            dst_clusters = cluster[edge_index[1]]

            pooled_edge_index = torch.stack([src_clusters, dst_clusters], dim=0)
            linear_idx = pooled_edge_index[0] * num_clusters + pooled_edge_index[1]

            adj_flat = torch.zeros(num_clusters * num_clusters, device=x.device)
            adj_flat = scatter_add(edge_weight, linear_idx, dim=0, out=adj_flat)

            pooled_adj = adj_flat.view(num_clusters, num_clusters)
            pooled_adj = (pooled_adj > 0).float()

            x_list.append(pooled_x)
            adj_list.append(pooled_adj)
            mask_list.append(torch.ones(num_clusters, dtype=torch.bool, device=x.device))

        # Pad pooled_x and pooled_adj to max clusters in batch for batching
        max_clusters = max(x_.size(0) for x_ in x_list)
        pooled_x_batch = torch.zeros((B, max_clusters, F), device=x.device)
        pooled_adj_batch = torch.zeros((B, max_clusters, max_clusters), device=x.device)
        pooled_mask_batch = torch.zeros((B, max_clusters), dtype=torch.bool, device=x.device)

        for b in range(B):
            n_c = x_list[b].size(0)
            pooled_x_batch[b, :n_c] = x_list[b]
            pooled_adj_batch[b, :n_c, :n_c] = adj_list[b]
            pooled_mask_batch[b, :n_c] = mask_list[b]

        return pooled_x_batch, pooled_adj_batch, pooled_mask_batch


class CrossChannelConv(nn.Module):
    """
    Cross channel convolution to refine multi-channel pooled representations.
    Simple example: 1D conv over concatenated features from channels.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # input_dim = num_channels * node_feature_dim
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv1d_out = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, N, C*F] -> transpose for conv1d: [B, C*F, N]
        x_t = x.transpose(1, 2)
        h = self.relu(self.conv1d(x_t))
        out = self.conv1d_out(h)
        out = out.transpose(1, 2)
        return out

class MuchPool(nn.Module):
    def __init__(self, node_feature_dim, ratio=0.6):
        super().__init__()
        self.ratio = ratio
        # Channel 1: dropping pooling on topology
        self.channel_topo = DropPoolChannel(node_feature_dim, ratio)

        # Channel 2: dropping pooling on node features
        self.channel_feat = DropPoolChannel(node_feature_dim, ratio)

        # Channel 3: coarsening pooling
        self.channel_coarse = CoarseningPoolChannel()

        # Cross channel convolution
        self.cross_conv = CrossChannelConv(input_dim=node_feature_dim*3, hidden_dim=node_feature_dim*3)

    def forward(self, x, adj, mask=None):
        # x: [B, N, F]
        # adj: [B, N, N]
        # mask: [B, N]

        # Channel 1: dropping based on topology - here, as proxy, use scores from node degrees
        deg = adj.sum(dim=-1)  # [B, N]
        deg = deg.unsqueeze(-1).repeat(1, 1, x.size(-1))  # [B, N, F]
        topo_scores = deg * x  # just a simple way to include topology in scoring
        pooled1_x, pooled1_adj, pooled1_mask = self.channel_topo(topo_scores, adj, mask)

        # Channel 2: dropping based on node features
        pooled2_x, pooled2_adj, pooled2_mask = self.channel_feat(x, adj, mask)

        # Channel 3: coarsening pooling
        pooled3_x, pooled3_adj, pooled3_mask = self.channel_coarse(x, adj, mask)

        # Pad all pooled node sets to same max nodes to combine
        max_nodes = max(pooled1_x.size(1), pooled2_x.size(1), pooled3_x.size(1))

        def pad_tensor(t, size):
            B, N, F = t.size()
            if N == size:
                return t
            padded = torch.zeros(B, size, F, device=t.device)
            padded[:, :N, :] = t
            return padded

        pooled1_x = pad_tensor(pooled1_x, max_nodes)
        pooled2_x = pad_tensor(pooled2_x, max_nodes)
        pooled3_x = pad_tensor(pooled3_x, max_nodes)

        # Concatenate features on feature dim (channels)
        concat_x = torch.cat([pooled1_x, pooled2_x, pooled3_x], dim=-1)  # [B, max_nodes, 3*F]

        # Cross channel conv refinement
        refined_x = self.cross_conv(concat_x)  # [B, max_nodes, 3*F]

        pooled_adj = pad_tensor(pooled3_adj, max_nodes)
        pooled_mask = torch.ones((x.size(0), max_nodes), dtype=torch.bool, device=x.device)

        return refined_x, pooled_adj, pooled_mask