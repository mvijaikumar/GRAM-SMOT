import numpy as np
import torch,pdb,math
from Evaluation import evaluate_model
from time import time
from Batch import Batch

class Valid_Test_Error_Bundle(object):
    def __init__(self,params):
        self.validNegativesDict   = params.validNegativesDict_bundle
        self.testNegativesDict    = params.testNegativesDict_bundle

        self.num_valid_instances  = params.num_valid_instances_bundle
        self.num_test_instances   = params.num_test_instances_bundle
        self.num_thread           = params.num_thread
        self.num_valid_negatives  = self.get_num_valid_negative_samples(self.validNegativesDict)
        self.valid_dim            = self.num_valid_negatives + 1

        self.epoch_mod            = 1
        self.params               = params
        self.valid_batch_siz      = params.valid_batch_siz
        self.at_k                 = params.at_k

        self.validArrTriplets,self.valid_pos_items = self.get_dict_to_triplets(self.validNegativesDict)
        self.testArrTriplets,self.test_pos_items   = self.get_dict_to_triplets(self.testNegativesDict)

    def get_num_valid_negative_samples(self,validDict):
        first_key = next(iter(validDict))
        return len(self.validNegativesDict[first_key])

    def get_dict_to_triplets(self,dct):
        user_lst, item_lst = [],[]
        pos_item_lst = []
        for key,value in dct.items():
            usr_id, itm_id, dom_id = key
            users  = list(np.full(self.valid_dim,usr_id,dtype = 'int32'))#+1 to add pos item
            items  = [itm_id]
            pos_item_lst.append(itm_id)
            items += list(value) # first is positive item

            user_lst   += users
            item_lst   += items

        bundle_lst = item_lst #item to bundle
        item_lst   = [0]*len(user_lst)
        return (np.array(user_lst),np.array(item_lst),np.array(bundle_lst)),np.array(pos_item_lst)

    def get_update(self,model,epoch_num,device,valid_flag=True):
        model.eval()
        if valid_flag == True:
            (user_input,item_input,bundle_input) = self.validArrTriplets
            num_inst   = self.num_valid_instances * self.valid_dim
            posItemlst = self.valid_pos_items # parameter for evaluate_model
            matShape   = (self.num_valid_instances, self.valid_dim)

        else:
            (user_input,item_input,bundle_input) = self.testArrTriplets
            num_inst   = self.num_test_instances * self.valid_dim
            posItemlst = self.test_pos_items # parameter for evaluate_model
            matShape   = (self.num_test_instances, self.valid_dim)

        batch_siz      = self.valid_batch_siz * self.valid_dim

        full_pred_torch_lst  = []
        user_input_ten    = torch.from_numpy(user_input.astype(np.long)).to(device)
        bundle_input_ten  = torch.from_numpy(bundle_input.astype(np.long)).to(device)
        batch             = Batch(num_inst,batch_siz,shuffle=False)
        while batch.has_next_batch():
            batch_indices = batch.get_next_batch_indices()
            y_pred        = model(user_input_ten[batch_indices],None,bundle_input_ten[batch_indices],None,None,'user-bundle')

            full_pred_torch_lst.append(y_pred)
        full_pred_np = torch.cat(full_pred_torch_lst).data.cpu().numpy()
        # ==============================

        predMatrix     = np.array(full_pred_np).reshape(matShape)
        itemMatrix     = np.array(bundle_input).reshape(matShape)

        (hits, ndcgs, maps) = evaluate_model(posItemlst=posItemlst,itemMatrix=itemMatrix,predMatrix=predMatrix,k=self.at_k,num_thread=self.num_thread)
        return (hits, ndcgs, maps)

