import numpy as np
import scipy.sparse as sp
from time import time
from Dataset import Dataset
import subprocess as sub
import pdb

class BundleDataset(Dataset):
    def __init__(self,args):
        Dataset.__init__(self,args)
        t2 = time()

        if self.load_embedding_flag == True:
            self.feature_dim        = self.attr_dim
            self.bundle_attr_mat    = self.load_embed_as_mat      (self.path + ".bundle_embed_final")
            self.bundle_embed_dict  = self.load_embed_file_as_dict(self.path + ".bundle_embed.final")
            self.user_item_bundle_attr_mat = self.combine_attr_mat(self.user_item_attr_mat,self.bundle_attr_mat)

        self.trainArrQuadruplets_bundle  = self.load_rating_file_as_arraylist_bundle(self.path + ".user_bundle.train",train_flag=True)
        self.train_matrix_bundle         = self.load_rating_file_as_matrix_bundle(self.path + ".user_bundle.train")

        # to choose negative samples
        self.validNegativesDict_bundle   = self.load_file_as_dict(self.path + ".user_bundle.valid") #.valid.negative
        self.testNegativesDict_bundle    = self.load_file_as_dict(self.path + ".user_bundle.test") #.test.negative

        # bundle_item (train equivalent)
        self.bundle_item_mat             = self.load_bundle_item_matrix(self.embed_path + ".bundle_item")
        self.user_bundle_mat             = self.train_matrix_bundle

        t1 = time()

        self.user_item_bundle_partial_adjacency_mat = self.get_user_item_and_bundle_item_adjacency_matrix_sparse(self.adjacency_user_item_mat, self.bundle_item_mat)
        t1 = time()
        self.user_item_bundle_adjacency_mat         = self.get_user_item_bundle_and_user_bundle_adjacency_matrix_sparse(self.user_item_bundle_partial_adjacency_mat, self.user_bundle_mat)

        self.num_nodes   = self.user_item_bundle_adjacency_mat.shape[0]

        # generation-related
        ##self.generation = args.generation
        self.generation = False ##

        if self.generation:
            self.bundlegen_items_dct             = self.load_entity_entities_dct(self.embed_path + ".bundle_item.mat")
            self.user_bundlegen_dct              = self.load_entity_entity_dct(self.embed_path + ".user_bundle.gen.valid") #dct for mapping user-bundle for bundlegen
            self.user_bundlegen_mat              = self.load_user_bundlegen_matrix(self.embed_path + ".user_bundle.gen.valid") #with userids not bundleids

    def load_entity_entities_dct(self, filename):
        entity1_entities_dict = dict()
        with open(filename, "r") as f:
            line = f.readline()
            toks = len(line.strip().split("\t")) - 1
            while line  != None and line != "":
                toks     = line.strip().split("\t")
                ent1     = int(toks[0])

                entities = np.array([int(x) for x in toks[1:]])
                entity1_entities_dict[ent1] = entities
                line = f.readline()
        return entity1_entities_dict

    def load_entity_entity_dct(self, filename):
        entity1_entity2_dict = dict()
        with open(filename, "r") as f:
            line = f.readline()
            toks = len(line.strip().split("\t")) - 1
            while line  != None and line != "":
                toks     = line.strip().split("\t")
                ent1,ent2     = int(toks[0]),int(toks[1])

                entity1_entity2_dict[ent1] = ent2
                line = f.readline()
        return entity1_entity2_dict

    def load_user_bundlegen_matrix(self, filename):
        # Construct matrix
        mat     = sp.dok_matrix((self.num_user,self.num_user), dtype=np.int)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, _ = (int(arr[0]), int(arr[1]))
                mat[user, user] = 1 ## note that here it is user-user (so no more than one bundle can be added)
                line = f.readline()
        return mat.tolil()

    def load_bundle_item_matrix(self, filename):
        # Construct matrix
        mat     = sp.dok_matrix((self.num_bundle,self.num_item), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = (int(arr[0]), int(arr[1]))
                mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_user_item_and_bundle_item_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        t1 = time()
        num_row,num_col = (mat1.shape[0] + mat2.shape[0], mat1.shape[0] + mat2.shape[0])
        mat = sp.lil_matrix((num_row,num_col),dtype=np.int8)
        assert num_row == num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],0:mat1.shape[0]]                           = mat1.astype(np.int8)
        mat[self.num_user+self.num_item:,self.num_user:self.num_user+self.num_item] = mat2.astype(np.int8)
        mat[self.num_user:self.num_user+self.num_item,self.num_user+self.num_item:] = mat2.astype(np.int8).T
        return mat#.tocsr()

    def get_user_item_bundle_and_user_bundle_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        mat = mat1.tolil() #sp.lil_matrix((num_row,num_col),dtype=np.int8)

        mat[self.num_user+self.num_item:,0:self.num_user] = mat2.astype(np.int8).T
        mat[0:self.num_user,self.num_user+self.num_item:] = mat2.astype(np.int8)

        return mat#.tocsr()

