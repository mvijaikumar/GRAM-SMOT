import numpy as np
import matplotlib.pyplot as plt
import distutils.dir_util
import subprocess,os

class Error_plot(object):
    def __init__(self,save_flag=False,res_path=None,args_str=None,args=None,item_bundle_str=''):

        self.loss_list,self.recon_loss_list,self.reg_loss_list,self.ce_loss_list = [],[],[],[]
        self.valid_hr_list,self.test_hr_list             = [],[]
        self.valid_ndcg_list,self.test_ndcg_list         = [],[]
        self.valid_map_list,self.test_map_list         = [],[]
        self.save_flag = save_flag
        self.item_bundle_str = item_bundle_str
        if save_flag == True:
            self.path     = res_path + args.res_folder + '/' # folder to store the current analysis results
            self.args_str = args_str
            self.args     = args
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def append(self,loss,recon_loss,reg_loss,ce_loss,valid_hr,test_hr,valid_ndcg,test_ndcg,valid_map,test_map):
        self.loss_list.append(loss)
        self.reg_loss_list.append(reg_loss)
        self.recon_loss_list.append(recon_loss)
        self.ce_loss_list.append(ce_loss)

        self.valid_hr_list.append(valid_hr)
        self.test_hr_list.append(test_hr)
        self.valid_ndcg_list.append(valid_ndcg)
        self.test_ndcg_list.append(test_ndcg)
        self.valid_map_list.append(valid_map)
        self.test_map_list.append(test_map)

    def plot(self):
        iter_count = len(self.loss_list)
        self.iterations = range(iter_count)

        self.plot_train_error()
        self.plot_ndcg()
        self.plot_hr()
        self.plot_map()

    def plot_train_error(self):
        plt.clf()
        plt.plot(self.iterations, self.loss_list ,'r--',label='total loss')
        plt.plot(self.iterations, self.recon_loss_list, 'b--',label='recon loss')
        plt.plot(self.iterations, self.reg_loss_list, 'g--',label='reg loss')
        plt.plot(self.iterations,self.ce_loss_list,'c--',label='ce loss')

        plt.xlabel('Iteration')
        plt.ylabel('Error values')
        plt.title('Iteration vs training error')
        plt.legend()
        plt.grid(True)
        if self.save_flag:
            plt.savefig(self.path + self.args_str + '_train.png', bbox_inches='tight')
        else:
            plt.show()

    def plot_ndcg(self):
        plt.clf()
        plt.plot(self.iterations, self.valid_ndcg_list, 'b-',label='valid ndcg')
        plt.plot(self.iterations, self.test_ndcg_list, 'g-', label='test ndcg')
        plt.xlabel('Iteration')
        plt.ylabel('ndcg values')
        plt.title('Iteration vs ndcg')
        plt.legend()
        plt.grid(True)
        if self.save_flag:
            plt.savefig(self.path + self.args_str + '_ndcg.png', bbox_inches='tight')
        else:
            plt.show()

    def plot_hr(self):
        plt.clf()
        plt.plot(self.iterations, self.valid_hr_list, 'b-',label='valid hr')
        plt.plot(self.iterations, self.test_hr_list, 'g-',label='test hr')

        plt.xlabel('Iteration')
        plt.ylabel('hr values')
        plt.title('Iteration vs hr')
        plt.legend()
        plt.grid(True)
        if self.save_flag:
            plt.savefig(self.path + self.args_str + '_hr.png', bbox_inches='tight')
        else:
            plt.show()

    def plot_map(self):
        plt.clf()
        plt.plot(self.iterations, self.valid_map_list, 'b-',label='valid map')
        plt.plot(self.iterations, self.test_map_list, 'g-',label='test map')

        plt.xlabel('Iteration')
        plt.ylabel('map values')
        plt.title('Iteration vs hr')
        plt.legend()
        plt.grid(True)
        if self.save_flag:
            plt.savefig(self.path + self.args_str + '_map.png', bbox_inches='tight')
        else:
            plt.show()

    def get_best_valid_test_error(self):
        best_valid_hr_index   = np.argmax(self.valid_hr_list)
        best_valid_ndcg_index = np.argmax(self.valid_ndcg_list)
        best_valid_map_index  = np.argmax(self.valid_map_list)
        best_valid_hr         = self.valid_hr_list[best_valid_hr_index]
        best_valid_ndcg       = self.valid_ndcg_list[best_valid_ndcg_index]
        best_valid_map        = self.valid_map_list[best_valid_map_index]
        best_test_hr          = self.test_hr_list[best_valid_hr_index]
        best_test_ndcg        = self.test_ndcg_list[best_valid_ndcg_index]
        best_test_map         = self.test_map_list[best_valid_map_index]

        self.args.hr_index,self.args.ndcg_index,self.args.map_index = best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index

        if self.item_bundle_str == 'bundle':
            self.args.hr_index_b,self.args.ndcg_index_b,self.args.map_index_b = best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index
        elif self.item_bundle_str == 'item':
            self.args.hr_index_i,self.args.ndcg_index_i,self.args.map_index_i = best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index

        if self.save_flag == True:
            result_file_append = 'result_bundle.res' if self.item_bundle_str == 'bundle' else 'result_item.res'
            print("Path where results are stored.",self.path + result_file_append)
            with open(self.path + result_file_append,'a') as fout:
                fout.write('val_hr: {:.4f} test_hr: {:.4f} val_ndcg: {:.4f} test_ndcg: {:.4f}  val_map: {:.4f} test_map: {:.4f} params: {}'.format(best_valid_hr,best_test_hr,best_valid_ndcg,best_test_ndcg,best_valid_map,best_test_map,str(self.args))+'\n')

        return best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index,best_valid_hr,best_valid_ndcg,best_valid_map,best_test_hr,best_test_ndcg,best_test_map
