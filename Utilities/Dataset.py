import numpy as np
import scipy.sparse as sp, pdb
from time import time
from collections import defaultdict

class Dataset(object):
    def __init__(self,args):
        t1 = time()
        dirname                  = args.path
        filename                 = args.dataset
        self.max_item_seq_length = args.max_item_seq_length
        self.method              = args.method

        self.path                = dirname + filename
        self.load_embedding_flag = True if args.load_embedding_flag == 1 else False

        path_valid_test          = dirname + filename

        ##self.embed_path          = self.get_embed_path(self.path,args.dataset)
        self.embed_path          = self.path
        self.num_user,self.num_item,self.num_dom     = self.get_user_item_count(self.path + ".user_item.train")
        if self.method in ['gram-smot']:
            self.num_user_bun,self.num_bundle,_        = self.get_user_item_count(self.path + ".user_bundle.train")
            self.num_bundle_bun, self.num_item_bun,_   = self.get_user_item_count(self.embed_path + ".bundle_item")
            self.num_user   = max(self.num_user,self.num_user_bun) ##num_user has to be checked
            self.num_item   = max(self.num_item,self.num_item_bun) ##num_user has to be checked
            self.num_bundle = max(self.num_bundle,self.num_bundle_bun) ##num_user has to be checked

            if self.load_embedding_flag == True:
                self.attr_dim            = self.get_item_embed_dim(self.path + self.item_embed_ext)
                self.user_attr_mat       = self.load_embed_as_mat      (self.path + ".user_embed_final")
                self.item_attr_mat       = self.load_embed_as_mat      (self.path + ".item_embed_final")
                self.user_embed_dict     = self.load_embed_file_as_dict(self.path + ".user_embed.final")
                self.item_embed_dict     = self.load_embed_file_as_dict(self.path + ".item_embed.final")
                self.user_item_attr_mat  = self.combine_attr_mat(self.user_attr_mat,self.item_attr_mat)

        self.trainArrQuadruplets = self.load_rating_file_as_arraylist(self.path + ".user_item.train",train_flag=True)
        (self.train_matrix, self.domain_matrix) = self.load_rating_file_as_matrix(self.path + ".user_item.train")

        # to choose negative samples
        self.validNegativesDict  = self.load_file_as_dict(path_valid_test + ".user_item.valid") #.valid.negative
        self.testNegativesDict   = self.load_file_as_dict(path_valid_test + ".user_item.test") #.test.negative
        t2 = time()

        # adjacency matrix
        self.adjacency_user_item_mat     = self.get_adjacency_matrix_sparse(self.train_matrix,self.train_matrix.T)
        self.num_nodes = self.adjacency_user_item_mat.shape[0]

    def get_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        num_row,num_col = (mat1.shape[0] + mat2.shape[0], mat1.shape[1] + mat2.shape[1])
        mat = sp.lil_matrix((num_row,num_col),dtype=np.int8)
        assert num_row == num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],mat1.shape[0]:] = mat1.astype(np.int8)#.tolil()
        mat[mat1.shape[0]:,0:mat1.shape[0]] = mat2.astype(np.int8)#.tolil()
        return mat#.tocsr()

    def combine_attr_mat(self, mat1, mat2):
        return np.concatenate([mat1,mat2],axis=0)

    def get_embed_path(self,path,dataset):
        path_words = path.replace('//','/').split('/')
        embed_path = '/'.join(path_words[:-3]) + '/' + dataset

        return embed_path

    def get_item_embed_dim(self,filename):
        item_embed = dict()
        with open(filename, "r") as f:
            line = f.readline().strip()
            toks = line.replace("\n","").split("::")
            itemid = int(toks[0])
            embed  = np.array(toks[1].split(" ")).astype(np.float)
            attr_dim = len(embed)
            return attr_dim

    def get_user_item_count(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline().strip()
            while line != None and line != "":
                arr         = line.split("\t")
                u, i      = int(arr[0]), int(arr[1])
                num_users   = max(num_users, u)
                num_items   = max(num_items, i)
                line = f.readline()
        return num_users+1, num_items+1, None

    def load_rating_file_as_arraylist(self, filename,train_flag=False):
        user_input, item_input, rating = [],[],[]

        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rat = (int(arr[0]), int(arr[1]), float(arr[2]))
                if rat > 0.0:
                    rat = 1.0
                user_input.append(user)
                item_input.append(item)
                rating.append(rat)
                line = f.readline()
        return np.array(user_input), np.array(item_input), np.array(rating,dtype=np.float16), None

    def load_embed_file_as_dict(self, filename):
        item_embed = dict()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks = line.replace("\n","").split("::")
                itemid = int(toks[0])
                embed  = np.array(toks[1].split(" ")).astype(np.float)
                item_embed[itemid] = embed
                line = f.readline()
        return item_embed

    def load_file_as_dict(self, filename):
        item_embed = dict()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks   = line.replace("\n","").split("::")
                keystr = toks[0].replace("(","").replace(")","").split(",")
                tup    = (int(keystr[0]),int(keystr[1]),int(keystr[2]))
                embed  = [int(x) for x in toks[1].split(" ")]
                item_embed[tup] = embed
                line = f.readline()
        return item_embed

    def load_rating_file_as_matrix(self, filename):
        # Construct matrix
        mat     = sp.dok_matrix((self.num_user,self.num_item), dtype=np.float32)
        dom_mat = sp.dok_matrix((self.num_user,self.num_item), dtype=np.int16)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, domain = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if (rating > 0):
                    mat[user, item] = 1.0
                    dom_mat[user,item] = domain + 10 # to avoid explicit zero elimination prob
                line = f.readline()
        return (mat.tolil(), dom_mat.tolil())

    def load_rating_file_as_matrix_bundle(self, filename):
        # Construct matrix
        mat     = sp.dok_matrix((self.num_user,self.num_bundle), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, domain = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat.tolil()

    def load_rating_file_as_arraylist_bundle(self, filename,train_flag=False):
        user_input, item_input, rating, domain = [],[],[],[]

        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rat, dom_num = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if rat > 0.0:
                    rat = 1.0
                user_input.append(user)
                item_input.append(item)
                rating.append(rat)
                domain.append(dom_num)
                line = f.readline()
        return np.array(user_input), np.array(item_input), np.array(rating,dtype=np.float16), np.array(domain)

    def load_rating_file_as_matrix_indiv(self, filename):
        # Construct matrix
        mat = dict()
        for ind in xrange(self.num_dom):
            mat[ind] = sp.dok_matrix((self.num_user,self.num_item), dtype=np.float32)

        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, dom = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if (rating > 0):
                    mat[dom][user, item] = 1.0
                line = f.readline()
        return mat

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_embed_as_mat(self, filename,flag='item'):
        # Construct matrix
        if flag == 'user':
            mat = np.zeros((self.num_user,self.attr_dim),dtype=np.float32)
        else:
            mat = np.zeros((self.num_itemidfiles * self.num_item,self.attr_dim),dtype=np.float32)
        ind =0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                ind = ind + 1
                toks    = line.replace("\n","").split("::")
                itemid  = int(toks[0])
                embed   = np.array(toks[1].split(" ")).astype(np.float)
                mat[itemid] = embed
                line = f.readline()
        return mat

    def get_item_count_domainwise(self, filename):
        pass
    def load_rating_file_as_matrix_train(self, filename):
        pass

    # attention related
    def get_target_domid(self,test_file):
        return int(open(test_file).readline().split('::')[0].split(',')[2].replace(')',''))

    def load_user_items_mat(self,filename,target_flag=True):
        if target_flag == True:
            user_items_dict = self.load_user_target_items_dict(filename)
            mat = np.full((self.num_user, self.max_item_seq_length),self.num_item) # last item index is allocated for padding, numof domains are multiplied for source domain
        else:
            user_items_dict = self.load_user_source_items_dict(filename)
            mat = np.full((self.num_user,self.num_dom * self.max_item_seq_length),self.num_item) # last item index is allocated for padding, numof domains are multiplied for source domain ##decide
        for user in user_items_dict.keys():
            for item in user_items_dict[user]:
                temp_shuff_arr = np.array(user_items_dict[user])
                if target_flag == True:
                    max_seq_len = min(self.max_item_seq_length, len(temp_shuff_arr))
                else:
                    max_seq_len = min(self.max_item_seq_length * self.num_dom, len(temp_shuff_arr))
                # to shuffle inplace
                np.random.shuffle(temp_shuff_arr)
                mat[user,0:max_seq_len] = temp_shuff_arr[:max_seq_len] # user_items_dict[user][:max_seq_len] ##shuffle selecting 20 (impt)

        return mat

    def load_user_source_items_dict(self,filename):
        return self.load_user_items_dict(filename,target_id=self.tar_dom,target_flag=False)

    def load_user_target_items_dict(self,filename):
        return self.load_user_items_dict(filename,target_id=self.tar_dom,target_flag=True)

    def load_user_items_dict(self,filename,target_id,target_flag=True):
        user_items_dict = defaultdict(list)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks = line.replace('\n','').split("\t")
                user, item, rating, dom = (int(toks[0]), int(toks[1]), float(toks[2]), int(toks[3]))
                if   (target_flag == True and dom == target_id):
                    user_items_dict[user].append(item)
                elif (target_flag == False and dom != target_id):
                    user_items_dict[user].append(item)
                line = f.readline()
        return user_items_dict

