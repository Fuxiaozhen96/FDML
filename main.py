import argparse
from GM_train import train
from GM_test import test
import numpy as np
import torch
import random

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', "--dataset", default='Corel16k001', type=str, help='dataset name')
parser.add_argument('-cp', "--checkpoint_path", default='./model/model_Corel16k001/lr-0.001_lr-decay_0.30_lr-times_4.0/FDML-1890', type=str, help='The path to a checkpoint from which to fine-tune.')
parser.add_argument('-bs', "--batch_size", default=256, type=int, help='the number of data points in one minibatch')
parser.add_argument('-tbs', "--test_batch_size", default=256, type=int, help='the number of data points in one testing or validation batch')
parser.add_argument('-lr', "--learning_rate", default=0.001, type=float, help='initial learning rate')
parser.add_argument('-epoch', "--max_epoch", default= 90, type=int, help='max epoch to train')
parser.add_argument('-wd', "--weight_decay", default=1e-5, type=float, help='weight decay rate')
parser.add_argument('-lrdr', "--lr_decay_ratio", default=0.3, type=float, help='The decay ratio of learning rate')
parser.add_argument('-lrdt', "--lr_decay_times", default=4.0, type=float, help='The number of times learning rate decays')
parser.add_argument('-seed', "--seed", default=0, type=int, help='seed')
parser.add_argument('-feat_dim', "--feature_dim", default=500, type=int, help='the dimensionality of the features')
parser.add_argument('-label_dim', "--label_dim", default=153, type=int, help='the number of labels in current training')
parser.add_argument('-latent_dim', "--latent_dim", default=155, type=int, help='the dimensionality of the latent features')
parser.add_argument('-keep_prob', "--keep_prob", default=0.1, type=float, help='drop out rate')
parser.add_argument('-lambda1', "--lambda1", default=3, type=float, help='lambda1')
parser.add_argument('-lambda2', "--lambda2", default=3, type=float, help='lambda2')
parser.add_argument('-lambda3', "--lambda3", default=0.1, type=float, help='lambda3')

parser.add_argument('-max_keep', "--max_keep", default=3, type=int, help='maximum number of saved model')
parser.add_argument('-check_freq', "--check_freq", default=40, type=int, help='checking frequency')
parser.add_argument('-T0', "--T0", default=200, type=int, help='optimizer T0')
parser.add_argument('-T_mult', "--T_mult", default=3, type=int, help='T_mult')
parser.add_argument('-eta_min', "--eta_min", default=2e-4, type=float, help='eta min')

parser.add_argument('-scale_coeff', "--scale_coeff", default=1.0, type=float, help='mu/logvar scale coefficient')
parser.add_argument('-mode', "--mode", default='test', type=str, help='latent reg')
parser.add_argument('-resume', "--resume", action='store_true', help='whether to resume a ckpt')

args = parser.parse_args()
args.train_idx='./data/'+args.dataset+'/'+args.dataset+'_train_data.mat'
args.valid_idx = './data/'+args.dataset+'/'+args.dataset+'_val_data.mat'
args.test_idx = './data/'+args.dataset+'/'+args.dataset+'_test_data.mat'

if __name__ == "__main__":
    np.random.seed(args.seed)  # set the random seed of numpy
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError("mode %s is not supported." % args.mode)
