import argparse
import sys
def parse_args():
    # dataset and method
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--method', nargs='?', default='gram-smot',help='gram-smot')
    parser.add_argument('--path', nargs='?',
                        default='./data/youshu/',help   ='Input data path.')
    parser.add_argument('--dataset', nargs='?',
                        default='youshu',          help='Choose a dataset.')
    parser.add_argument('--res_path',              nargs='?',default='~/BundleRec_Data/result/',help='result path for plots and best error values.')
    parser.add_argument('--res_folder',            nargs='?',default='',help='specific folder corresponding to different runs on different parameters.')
    parser.add_argument('--flag_types',            nargs='?',  default="['user-item','bundle-item','user-bundle']",help='loss based on the given interactions.')

    # algo-parameters
    parser.add_argument('--epochs',                type=int, default=200,help='Number of epochs.')
    parser.add_argument('--batch_size',            type=int, default=2048,help='Batch size.')
    parser.add_argument('--valid_batch_siz',       type=int, default=2048,help='Valid batch size.')
    parser.add_argument('--lr',                    type=float, default=.0007,help='Learning rate.')
    parser.add_argument('--initializer',           nargs='?', default='xavier',help='xavier')
    parser.add_argument('--stddev',                type=float, default=0.02,help='stddev for normal and [min,max] for uniform')
    parser.add_argument('--optimizer',             nargs='?', default='adam',help='adam')
    parser.add_argument('--loss',                  nargs='?', default='ce',help='ce')

    # hyper-parameters
    parser.add_argument('--num_factors',           type=int, default=64,  help='Embedding size.')
    parser.add_argument('--num_layers',            type=int, default=2,    help='Number of hidden layers.') # feature in testing ##not completed
    parser.add_argument('--num_negatives',         type=int, default=7, help='Negative instances in sampling.')
    parser.add_argument('--reg_Wh',                type=float, default=0.0000, help="Regularization for weight vector.")
    parser.add_argument('--reg_bias',              type=float, default=0.000,help="Regularization for user and item bias embeddings.")
    parser.add_argument('--reg_lambda',            type=float, default=0.000,help="Regularization lambda for user and item embeddings.")
    parser.add_argument('--keep_prob',             type=float, default=0.7, help='droupout keep probability in layers.')
    parser.add_argument('--max_item_seq_length',   type=int, default=25, help='number of rated items to keep.') #20
    parser.add_argument('--load_embedding_flag',   type=int, default=0, help='0-->donot load embedding, 1-->load embedding for entities.')
    parser.add_argument('--margin',                type=float, default=2.0, help='margin value for TripletMarginLoss.')

    # misc
    parser.add_argument('--dataset_avg_flag_zero', type=int, default=0,  help='Dataset item embed zero (or) avg. zero --> 1, else avg')
    parser.add_argument('--attention_keep',        type=float, default=0.4, help='proj keep probability in projection weights layers for reviews.')

    # gat
    parser.add_argument('--neighbourhood_dp_keep', type=float, default=0.7, help='dropout keep prob. on neighbour nodes.')
    parser.add_argument('--proj_keep',             type=float, default=0.7, help='proj keep probability in projection weights layers for reviews.')
    parser.add_argument('--hid_units',             nargs='?',  default='[64,32]',help='hidden units of GAT')
    parser.add_argument('--n_heads',               nargs='?',  default='[4,2]',help='number of heads of GAT')

    # valid and test
    parser.add_argument('--at_k',                  type=int,  default=5,help='@k for recall, map and ndcg, etc.')
    parser.add_argument('--num_thread',            type=int,  default=16,help='number of threads.')
    parser.add_argument('--comment',               nargs='?', default='comment',help='comments about the current experimental iterations.')

    return parser.parse_args()
