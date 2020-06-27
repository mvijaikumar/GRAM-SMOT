import numpy as np
import sys,pdb
from time import time
import scipy.sparse as sp

class NegativeSamples(object):
    def __init__(self, sp_matrix, num_negatives, loss_criterion='ce'):
        # param assignment
        self.sp_matrix        = sp_matrix
        self.num_negatives    = num_negatives
        self.loss_criterion   = loss_criterion
        self.num_rating       = sp_matrix.nnz
        self.num_items        = sp_matrix.shape[-1] ##

        # positive part
        self.user_pos_arr,self.item_pos_arr,self.rating_pos_arr = self.get_positive_instances(sp_matrix)

        # negative part
        self.user_neg_arr   = np.repeat(self.user_pos_arr,self.num_negatives) ##negative samples could be different bw item and bundle
        self.rating_neg_arr = np.repeat([0],len(self.rating_pos_arr) * self.num_negatives)

        # positive_and_negative part pre-generated to improve efficiency
        self.user_arr   = np.concatenate([self.user_pos_arr,self.user_neg_arr])
        self.rating_arr = np.concatenate([self.rating_pos_arr,self.rating_neg_arr])
        self.rating_arr = self.rating_arr.astype(np.float16)

    def get_positive_instances(self,mat):
        user_pos_arr,item_pos_arr,rating_pos_arr=(np.array([],dtype=np.int),np.array([],dtype=np.int),np.array([],dtype=np.int))
        pos_mat = mat.tocsc().tocoo()
        user_pos_arr,item_pos_arr = pos_mat.row,pos_mat.col
        rating_pos_arr = np.repeat([1],len(user_pos_arr))

        return user_pos_arr,item_pos_arr,rating_pos_arr

    def generate_negative_item_samples(self,):
        neg_item_arr = np.array([],dtype=np.int)
        if self.loss_criterion == 'pairwise':
            random_indices = np.random.choice(self.num_items, 1 * self.num_rating)
        else:
            random_indices = np.random.choice(self.num_items, self.num_negatives * self.num_rating)
        neg_item_arr = random_indices

        return neg_item_arr

    # call this function from outside to generate instances at each epochs
    def generate_instances(self,):
        self.item_neg_arr = self.generate_negative_item_samples()
        self.item_arr     = np.concatenate([self.item_pos_arr,self.item_neg_arr])
        return self.user_arr,self.item_arr,self.rating_arr

    def generate_instances_bpr(self,):
        self.item_neg_arr = self.generate_negative_item_samples()
        return self.user_pos_arr,self.item_pos_arr,self.item_neg_arr
