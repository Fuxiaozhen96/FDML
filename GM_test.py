import numpy as np
import torch
import sys
import datetime
from copy import deepcopy
import evals
from utils import build_path, get_label, get_feat,get_poex_naex, THRESHOLDS
from GM_CVAE import FDML, compute_loss
from numpy import loadtxt
import scipy.io as scio
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

def test(args):
    METRICS = ['RL','OE','CV','AP','HL']
    print('reading npy...')

    test_feature=np.array(scio.loadmat(args.test_idx)['feature'],dtype='float32')
    test_label=np.array(scio.loadmat(args.test_idx)['label'],dtype='int')
    test_idx = [i for i in range(len(test_feature))]# load the indices of the training set

    print('reading completed')
    label_prototypical=0
    print('building network...')
    model = FDML(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    print("loaded model: %s" % (args.checkpoint_path))

    def test_step(test_idx):

        all_total_loss = 0
        all_pred_x = []
        all_label = []
        all_indiv_max = []
        all_feat_mu = []
        all_label_mu = []

        real_batch_size = min(args.batch_size, len(test_idx))

        N_test_batch = int((len(test_idx) - 1) / real_batch_size) + 1

        for i in range(N_test_batch):
            if i % 20 == 0:
                print("%.1f%% completed" % (i * 100.0 / N_test_batch))

            start = real_batch_size * i
            end = min(real_batch_size * (i + 1), len(test_idx))

            input_feat=test_feature[test_idx[start:end],:]
            input_label = test_label[test_idx[start:end],:]

            input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
            input_label = deepcopy(input_label).float().to(device)

            input_label = deepcopy(input_label).float().to(device)


            with torch.no_grad():
                y_z_mu, y_z_logvar, y_z, x_z_mu, x_z_logvar, x_z, x_zs_mu, x_zs_logvar, xz_y, yz_x ,z_y= model(input_feat,input_label, mode='test')
                total_loss = compute_loss(y_z_mu, y_z_logvar, y_z, x_z_mu, x_z_logvar, x_z, x_zs_mu, x_zs_logvar, xz_y,
                                          yz_x, z_y, input_feat, input_label, args.lambda1, args.lambda2, args.lambda3)
                pred_x=xz_y

            all_total_loss += total_loss * (end - start)

            if (all_pred_x == []):
                all_pred_x = pred_x.cpu().data.numpy()
                all_label = input_label.cpu().data.numpy()

            else:
                all_pred_x = np.concatenate((all_pred_x, pred_x.cpu().data.numpy()))
                all_label = np.concatenate((all_label, input_label.cpu().data.numpy()))

        total_loss = all_total_loss / len(test_idx)
        return all_pred_x, all_label, all_feat_mu, all_label_mu

    pred_x, input_label, all_feat_mu, all_label_mu = test_step(test_idx)

    best_test_metrics = None
    for threshold in THRESHOLDS:
        test_metrics = evals.compute_metrics(pred_x, input_label, threshold, all_metrics=True)
        if best_test_metrics == None:
            best_test_metrics = {}
            for metric in METRICS:
                best_test_metrics[metric] = test_metrics[metric]

    print("****************")
    for metric in METRICS:
        print(metric, ":", best_test_metrics[metric])
    print("****************")
