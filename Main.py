import sys,math,argparse,os,pdb,random
import numpy as np
import torch
from time import time

sys.path.append('./.')
sys.path.append('./Utilities/.')
sys.path.append('./comparison/.')
sys.path.append('./Models/.')

from Arguments import parse_args
from NegativeSamples import NegativeSamples
from Dataset import Dataset
from BundleDataset import BundleDataset
from Parameters import Parameters
from Valid_Test_Error_Bundle import Valid_Test_Error_Bundle
from Valid_Test_Error_Item import Valid_Test_Error_Item
from Evaluation import evaluate_model
from Error_plot import Error_plot
from Models import Models
from Batch import Batch
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn

if __name__ == '__main__':

    args = parse_args()
    print(args)
    print('Data loading...')
    t1,t_init = time(),time()
    dataset = BundleDataset(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = Parameters(args,dataset)
    print("""Load data done [%.1f s]. #user:%d, #item:%d, #train:%d, #test:%d, #valid:%d"""% (time() - t1, params.num_users,
        params.num_items,params.num_train_instances,params.num_test_instances,params.num_valid_instances))
    print("""Load data done [%.1f s]. #bundle:%d, #bundle_train:%d, #bundle_test:%d, #bundle_valid:%d"""% (time() - t1, params.num_bundles,
        params.num_train_instances_bundle,params.num_test_instances_bundle,params.num_valid_instances_bundle))

    args.args_str      = params.get_args_to_string()
    t1 = time()

    # model-loss-optimizer defn =======================================================================
    models         = Models(params)
    model          = models.get_model()

    criterion_ub   = torch.nn.BCELoss()
    criterion_ui   = torch.nn.TripletMarginLoss(margin=params.margin,p=2)
    criterion_bi   = torch.nn.TripletMarginLoss(margin=params.margin,p=2)

    optimizer_ub     = torch.optim.Adam(model.parameters(), lr=params.learn_rate)
    optimizer_ui     = torch.optim.Adam(model.parameters(), lr=params.learn_rate/10)
    optimizer_bi     = torch.optim.Adam(model.parameters(), lr=params.learn_rate/10)

    model.to(device)

    # training =======================================================================
    vt_err_bundle = Valid_Test_Error_Bundle(params)
    error_plot_bundle = Error_plot(save_flag=True,res_path=params.result_path,args_str=args.args_str,args=args,item_bundle_str='bundle')

    ns_ub = NegativeSamples(params.train_matrix_bundle,params.num_negatives)
    ns_ui = NegativeSamples(params.train_matrix,params.num_negatives,'pairwise')
    ns_bi = NegativeSamples(params.bundle_item_mat,params.num_negatives,'pairwise')

    flag_types = eval(args.flag_types)
    for epoch_num in range(params.num_epochs+1):
        tt = time()
        model.train()
        for flag_type in flag_types:
            vt_err     =  vt_err_bundle
            if flag_type == 'user-bundle':
                user_input,bundle_input,train_rating   = ns_ub.generate_instances()
                user_input,bundle_input,train_rating   = torch.from_numpy(user_input.astype(np.long)).to(device),torch.from_numpy(bundle_input.astype(np.long)).to(device),torch.from_numpy(train_rating.astype(np.float32)).to(device)
                num_inst = len(user_input)
            elif flag_type == 'user-item':
                user_input,item_input,item_input_neg   = ns_ui.generate_instances_bpr()
                user_input,item_input,item_input_neg   = torch.from_numpy(user_input.astype(np.long)).to(device),torch.from_numpy(item_input.astype(np.long)).to(device),torch.from_numpy(item_input_neg.astype(np.long)).to(device)
                num_inst = len(user_input)
            elif flag_type == 'bundle-item':
                bundle_input,item_input,item_input_neg = ns_bi.generate_instances_bpr()
                bundle_input,item_input,item_input_neg = torch.from_numpy(bundle_input.astype(np.long)).to(device),torch.from_numpy(item_input.astype(np.long)).to(device),torch.from_numpy(item_input_neg.astype(np.long)).to(device)
                num_inst = len(bundle_input)

            # ===============================
            t2 = time()
            ce_or_pairwise_loss, reg_loss, recon_loss   = 0.0, 0.0, 0.0

            batch    = Batch(num_inst,params.batch_size,shuffle=True) ##True --> batch can be kept common; do not reinitialize the batch
            if flag_type == 'user-bundle':
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_ub.zero_grad()

                    y_pred = model(user_input[batch_indices],None, bundle_input[batch_indices],
                                   None,None,flag_type)
                    y_orig = train_rating[batch_indices]
                    loss = criterion_ub(y_pred,y_orig)
                    loss.backward()
                    optimizer_ub.step()
                    ce_or_pairwise_loss += loss * len(batch_indices)
            elif flag_type == 'user-item':
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_ui.zero_grad()

                    user_embeds_batch, item_embeds_batch, item_embeds_neg_batch = model(user_input[batch_indices],item_input[batch_indices],
                                                                                        None, item_input_neg[batch_indices],None,flag_type)
                    loss = criterion_ui(user_embeds_batch, item_embeds_batch, item_embeds_neg_batch)
                    loss.backward()
                    optimizer_ui.step()
                    ce_or_pairwise_loss += loss * len(batch_indices)
            elif flag_type == 'bundle-item':
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_bi.zero_grad()

                    bundle_embeds_batch, item_embeds_batch, item_embeds_neg_batch = model(None,item_input[batch_indices], bundle_input[batch_indices],
                                                                                          item_input_neg[batch_indices],None, flag_type)
                    loss = criterion_bi(bundle_embeds_batch, item_embeds_batch, item_embeds_neg_batch)
                    loss.backward()
                    optimizer_bi.step()
                    ce_or_pairwise_loss += loss * len(batch_indices) ## ce_loss should be changed in appropriate place to include tripletloss

            total_loss = ce_or_pairwise_loss + reg_loss + recon_loss
            print("""[%.2f s] %15s iter:%3i obj ==> total loss:%.4f ce/pairwise loss:%.4f reg loss:%.4f recon loss:%.4f """
                %(time()-t2,flag_type, epoch_num,total_loss,ce_or_pairwise_loss,reg_loss,recon_loss))

        # validation and test =======================================================================
        t3 = time()
        (valid_hits_lst,valid_ndcg_lst,valid_map_lst) = vt_err.get_update(model,epoch_num,device,valid_flag=True)
        (test_hits_lst,test_ndcg_lst,test_map_lst)    = vt_err.get_update(model,epoch_num,device,valid_flag=False)
        (valid_hr,valid_ndcg,valid_map)               = (np.mean(valid_hits_lst),np.mean(valid_ndcg_lst),np.mean(valid_map_lst))
        (test_hr,test_ndcg,test_map)                  = (np.mean(test_hits_lst),np.mean(test_ndcg_lst),np.mean(test_map_lst))
        print("[%.2f s] %15s Errors train %.4f valid hr: %.4f test hr: %.4f valid ndcg: %.4f test ndcg: %.4f valid map: %.4f test map: %.4f"%(time()-t3,'',ce_or_pairwise_loss/num_inst,valid_hr,test_hr,valid_ndcg,test_ndcg,valid_map,test_map))

        error_plot_bundle.append(loss,recon_loss,reg_loss,ce_or_pairwise_loss,valid_hr,test_hr,valid_ndcg,test_ndcg,valid_map,test_map)

        print('Time taken for this epoch: {:.2f} m'.format((time()-tt)/60))

    # best valid and test =======================================================================
    tot_time = time() - t_init
    args.total_time = '{:.2f}m'.format(tot_time/60)
    print('error plot: ')
    for bundle_flag in [True]:
        (best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index,best_valid_hr,best_valid_ndcg,best_valid_map,best_test_hr,best_test_ndcg,best_test_map) = error_plot_bundle.get_best_valid_test_error()
        args.hr_index_bundle,args.ndcg_index_bundle,args.map_index_bundle = best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index
        print('[{:.2f} s] [Bundle] best_hr_index: {} best_ndcg_index: {} best_map_index: {} best_valid_hr: {:.4f} best_valid_ndcg: {:.4f} best_valid_map: {:.4f} best_test_hr: {:.4f} best_test_ndcg: {:.4f} best_test_map: {:.4f}'.format(tot_time,best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index,best_valid_hr,best_valid_ndcg,best_valid_map,best_test_hr,best_test_ndcg,best_test_map))
        error_plot_bundle.plot()
