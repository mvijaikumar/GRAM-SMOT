import sys,pdb
import numpy as np
import scipy.sparse as sp
import random,pickle

class Parameters(object):
    def __init__(self,args,dataset):
        # method ==============================================================
        self.method               = args.method.lower()
        self.args                 = args
        self.result_path          = args.res_path + args.dataset + '/' + args.method + '/'
        self.load_embedding_flag  = dataset.load_embedding_flag #indicates extra-information
        self.loss                 = args.loss
        self.path                 = args.path

        self.proj_keep = args.proj_keep
        self.neighbourhood_dp_keep = args.neighbourhood_dp_keep
        # GRAM-SMOT ==============================================================
        self.hid_units = eval(args.hid_units)
        self.n_heads   = eval(args.n_heads)

        # count ===============================================================
        self.num_users            = dataset.num_user
        self.num_items            = dataset.num_item
        self.num_train_instances  = dataset.train_matrix.nnz ##len(dataset.trainArrQuadruplets[0])
        self.num_valid_instances  = len(dataset.validNegativesDict.keys())
        self.num_test_instances   = len(dataset.testNegativesDict.keys())
        self.num_nodes            = dataset.num_nodes

        # data-structures ======================================================
        self.train_matrix         = dataset.train_matrix
        self.testNegativesDict    = dataset.testNegativesDict
        self.validNegativesDict   = dataset.validNegativesDict

        ## new
        self.adjacency_user_item_mat = dataset.adjacency_user_item_mat

        # count and data-structure for bundles ================================
        if self.method in ['gram-smot']:
            self.num_train_instances_bundle  = dataset.train_matrix_bundle.nnz ##len(dataset.trainArrQuadruplets[0])
            self.num_valid_instances_bundle  = len(dataset.validNegativesDict_bundle.keys())
            self.num_test_instances_bundle   = len(dataset.testNegativesDict_bundle.keys())
            self.num_bundles                 = dataset.num_bundle

            self.train_matrix_bundle         = dataset.train_matrix_bundle
            self.testNegativesDict_bundle    = dataset.testNegativesDict_bundle
            self.validNegativesDict_bundle   = dataset.validNegativesDict_bundle

            self.user_item_bundle_adjacency_mat = dataset.user_item_bundle_adjacency_mat
            self.bundle_item_mat  = dataset.bundle_item_mat

        # item_attr_related ====================================================
        if self.load_embedding_flag  == True:
            self.attr_dim  = dataset.attr_dim
            if self.method in ['gram-smot']:
                self.user_item_attr_mat        = dataset.user_item_attr_mat
                self.user_item_bundle_attr_mat = dataset.user_item_bundle_attr_mat

        # algo-parameters =======================================================
        self.num_epochs      = args.epochs
        self.batch_size      = args.batch_size
        self.valid_batch_siz = args.valid_batch_siz
        self.learn_rate      = args.lr
        self.optimizer       = args.optimizer
        self.proj_keep       = args.proj_keep
        self.attention_keep  = args.attention_keep

        # valid test ============================================================
        self.at_k            = args.at_k
        self.num_thread      = args.num_thread

        # hyper-parameters ======================================================
        self.num_factors     = args.num_factors
        self.num_layers      = args.num_layers ## testing
        self.num_negatives   = args.num_negatives
        self.reg_w           = args.reg_Wh
        self.reg_b           = args.reg_bias
        self.reg_lam         = args.reg_lambda
        self.keep_prob       = args.keep_prob
        self.max_item_seq_length = args.max_item_seq_length
        self.margin          = args.margin

        # generation ========================================================
        #self.generation                   = args.generation
        self.generation = False ##
        if self.generation:
            self.num_bundlegens           = self.num_users #not bundles
            self.bundlegen_items_dct      = dataset.bundlegen_items_dct
            self.user_bundlegen_dct       = dataset.user_bundlegen_dct
            self.user_bundlegen_mat       = dataset.user_bundlegen_mat

            #pdb.set_trace()
            self.num_negative_items_gen   = args.num_negative_items_gen
            self.num_items_per_bundlegen  = len(self.bundlegen_items_dct[0]) - self.num_negative_items_gen ## 0 may not present, get first value in other way
            self.user_item_bundle_bundlegen_mat = self.get_adjacency_matrix_sparse(self.user_item_bundle_adjacency_mat,self.user_bundlegen_mat)

    def dct_to_sparse_matrix_creation(self,user_to_bundles_dct,num_users,num_bundles): # some implementation but
        user_indices = []
        bundle_indices = []
        bundles_per_user = len(user_to_bundles_dct[0]) ## 0 may not be present==> in that case get first key and corr. value to get the length
        for user in user_to_bundles_dct:
            user_indices.append(np.repeat(user,bundles_per_user))
            bundle_indices.append(user_to_bundles_dct[user])

        user_indices_arr = np.concatenate(user_indices)
        bundle_indices_arr = np.concatenate(bundle_indices)
        values = np.repeat(1, len(user_indices_arr))

        user_to_bundles_spmat = sp.coo_matrix((values,(user_indices_arr,bundle_indices_arr)),shape=(num_users,num_bundles))
        return user_to_bundles_spmat.tolil()

    def get_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        num_row,num_col = (mat1.shape[0] + mat2.shape[1], mat1.shape[1] + mat2.shape[1])
        mat = sp.lil_matrix((num_row,num_col),dtype=np.int8)
        assert num_row == num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],0:mat1.shape[0]] = mat1.astype(np.int8)#.tolil()
        mat[0:mat2.shape[0],mat1.shape[1]:]  = mat2.astype(np.int8)#.tolil()
        mat[mat1.shape[0]:,0:mat2.shape[0]] = mat2.astype(np.int8).T#.tolil()
        return mat#.tocsr()

    def get_args_to_string(self,):
        args_str = str(random.randint(1,1000000))
        return args_str

    def get_optimizer(self,lr,optimizer='rmsprop'):
        pass

    def aggregate_items_to_bundle(self,):
        pass

    # ============================
    def get_performance_for_generation_task(self, num_pos_items, user_bundlegen_dct, bundlegen_items_dct, resultant_userbundle_tuple_itemset_dct, k):
        prec_k_lst, recall_k_lst, f1_k_lst, map_k_lst = [], [], [], []
        for user in user_bundlegen_dct:
            bundle = user_bundlegen_dct[user]
            user_bundle_tuple = (user,bundle)

            assert k <= num_pos_items, "k should be less than or equal to number of positive items in extracted bundles."
            true_labels = list(bundlegen_items_dct[bundle][0:k])
            pred_labels = resultant_userbundle_tuple_itemset_dct[user_bundle_tuple][0:k]

            prec_k,recall_k,f1_k = self.get_performance_at_k(true_labels=true_labels, pred_labels=pred_labels, k=k)
            map_k_lst.append(self.get_map_at_k(true_labels=true_labels, pred_labels=pred_labels, k=k))
            prec_k_lst.append(prec_k)
            recall_k_lst.append(recall_k)
            f1_k_lst.append(f1_k)
        n = len(prec_k_lst)
        return np.sum(prec_k_lst)/n, np.sum(recall_k_lst)/n, np.sum(f1_k_lst)/n, np.sum(map_k_lst)/n

    def get_performance_at_k(self,true_labels,pred_labels,k): # k should be 5 and true items in bundles should be 10, prec@5,recall@5,map@5,f1-score@5
        num_correct            = len(set(true_labels).intersection(set(pred_labels[0:k])))
        prec_at_k, recall_at_k = num_correct/k, num_correct/len(true_labels)
        f1_score_at_k          = 0.0 if (prec_at_k + recall_at_k) <= 0.0 else (2.0 * prec_at_k * recall_at_k ) /(prec_at_k + recall_at_k)
        return prec_at_k, recall_at_k, f1_score_at_k

    def get_recall_at_k(self,true_labels,pred_labels,k): # for testing
        return len(set(true_labels).intersection(set(pred_labels[0:k])))/len(true_labels)

    def get_map_at_k(self,true_labels, pred_labels, k):
        map_val = 0.0
        j = 0
        for i in range(len(pred_labels[0:k])):
            item = pred_labels[i]
            if item in true_labels:
                map_val += (1.0+j) / (i+1)
                j += 1
        return map_val/k 
