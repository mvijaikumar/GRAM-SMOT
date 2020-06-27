from __future__ import print_function
import torch, pdb,sys
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

sys.path.append('./pygat/.')
from utils import normalize_adj

from models import GAT, SpGAT
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv  # noqa

class GRAMSMOT(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(GRAMSMOT,self).__init__()

        self.device                      = device
        self.params                      = params
        self.user_item_bundle_embeddings = torch.nn.Embedding(params.num_nodes,params.num_factors).to(device)
        self.fc1                         = nn.Linear(params.num_factors,int(params.num_factors))
        self.fc2                         = nn.Linear(int(params.num_factors),1)
        self.fc3                         = nn.Linear(int(params.num_factors),1)
        self.dropout_user                = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout_item                = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout_bundle              = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob

        self.dropout4                    = torch.nn.Dropout(1.0 - params.proj_keep) ## keep_prob
        self.dropout5                    = torch.nn.Dropout(1.0 - params.proj_keep) ## keep_prob

        self.sigmoid                     = torch.nn.Sigmoid()
        self.elu                         = torch.nn.ELU()

        self.num_users,self.num_items,self.num_bundles = params.num_users,params.num_items, params.num_bundles

	# GAT =================================
        self.conv1                       = GATConv(in_channels=params.num_factors,
							out_channels=params.hid_units[0],
							heads=params.n_heads[0],
							dropout=1.0-params.neighbourhood_dp_keep)
        self.conv2                       = GATConv(in_channels=params.hid_units[0]*params.n_heads[0],
							out_channels=params.hid_units[1],
							heads=params.n_heads[1],
							dropout=1.0-params.neighbourhood_dp_keep)

        self.fc_gat                      = nn.Linear(params.hid_units[-1]*params.n_heads[-1],1)

        self.all_indices                 = torch.tensor(np.array(range(params.num_nodes))).to(device)

        # ==============================
        torch.nn.init.xavier_uniform_(self.user_item_bundle_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.sigmoid                     = torch.nn.Sigmoid()
        self.all_indices                 = torch.tensor(np.array(range(params.num_nodes))).to(device)

        self.adj                         = params.user_item_bundle_adjacency_mat.tocoo()
        self.adj                         = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        self.edge_indices                     = torch.LongTensor(self.adj.nonzero()).to(device)

    def forward(self, user_indices, item_indices, bundle_indices,
                      item_indices_negative, bundle_indices_negative, flag_type='user-item',
                      add_bundle=False, add_bundle_first_time=False):
        self.user_item_bundle_embeddings_weight = self.user_item_bundle_embeddings(self.all_indices)
        if True:
            x = self.user_item_bundle_embeddings_weight
            x = self.elu(self.conv1(x, self.edge_indices))
            x = self.dropout5(x)
            x = self.conv2(x, self.edge_indices)
            self.user_item_bundle_embeddings_gnn = x
        else:
            self.user_item_bundle_embeddings_gnn = self.user_item_bundle_embeddings_weight

        if flag_type == 'user-item':
            self.user_embeds                = self.user_item_bundle_embeddings_gnn[user_indices]#.view((1, -1))
            self.user_embeds_dp             = self.dropout_user(self.user_embeds)
            self.item_embeds                = self.user_item_bundle_embeddings_gnn[self.params.num_users+item_indices]
            self.item_embeds_dp             = self.dropout_item(self.item_embeds)
            self.item_embeds_neg            = self.user_item_bundle_embeddings_gnn[self.params.num_users+item_indices_negative]
            self.item_embeds_neg_dp         = self.dropout_item(self.item_embeds_neg)
            return self.user_embeds_dp, self.item_embeds_dp, self.item_embeds_neg_dp

        elif flag_type == 'user-bundle':
            self.user_embeds                = self.user_item_bundle_embeddings_gnn[user_indices]#.view((1, -1))
            self.user_embeds_dp             = self.dropout_user(self.user_embeds)
            self.bundle_embeds              = self.user_item_bundle_embeddings_gnn[self.params.num_users+self.params.num_items + bundle_indices]
            self.bundle_embeds_dp           = self.dropout_bundle(self.bundle_embeds)
            self.user_bundle_interactions   = self.user_embeds_dp * self.bundle_embeds_dp
            rating_pred                     = self.sigmoid(self.fc_gat(self.user_bundle_interactions).reshape(-1))
            return rating_pred

        elif flag_type == 'bundle-item':
            self.bundle_embeds              = self.user_item_bundle_embeddings_gnn[self.params.num_users+self.params.num_items + bundle_indices]
            self.bundle_embeds_dp           = self.dropout_bundle(self.bundle_embeds)
            self.item_embeds                = self.user_item_bundle_embeddings_gnn[self.params.num_users+item_indices]
            self.item_embeds_dp             = self.dropout_item(self.item_embeds)
            self.item_embeds_neg            = self.user_item_bundle_embeddings_gnn[self.params.num_users+item_indices_negative]
            self.item_embeds_neg_dp         = self.dropout_item(self.item_embeds_neg)
            return self.bundle_embeds_dp,self.item_embeds_dp,self.item_embeds_neg_dp

    def loss(self,):
        pass
