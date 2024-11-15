import math
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os
from copy import copy, deepcopy
import evals
from utils import build_path, THRESHOLDS,init_label_embed,gen_A,gen_adj
from GM_CVAE import FDML, compute_loss
import scipy.io as scio
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

METRICS=['RL','OE','CV','AP','HL']

def train(args):

    print('reading npy...')
    train_feature=np.array(scio.loadmat(args.train_idx)['feature'],dtype='float32')
    train_label=np.array(scio.loadmat(args.train_idx)['label'],dtype='int')
    val_feature=np.array(scio.loadmat(args.valid_idx)['feature'],dtype='float32')
    val_label=np.array(scio.loadmat(args.valid_idx)['label'],dtype='int')
    label_embed=init_label_embed(train_feature,train_label)
    adj=gen_A(train_label)
    A=gen_adj(adj)
    train_idx=[i for i in range(len(train_feature))]
    param_setting = "lr-{}_lr-decay_{:.2f}_lr-times_{:.1f}".format(args.learning_rate,args.lr_decay_ratio,args.lr_decay_times)
    build_path('summary/{}/{}'.format(args.dataset, param_setting))
    build_path('model/model_{}/{}'.format(args.dataset, param_setting))
    summary_dir = 'summary/{}/{}'.format(args.dataset, param_setting)
    model_dir = 'model/model_{}/{}'.format(args.dataset, param_setting)
    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)  # compute the number of iterations in each epoch
    n_iter = one_epoch_iter * args.max_epoch
    print("one_epoch_iter:", one_epoch_iter)
    print("total_iter:", n_iter)
    writer = SummaryWriter(log_dir=summary_dir)
    print('building network...')
    # building the model
    model = FDML(args).to(device)
    # log the learning rate
    writer.add_scalar('learning_rate', args.learning_rate)
    # use the RMSprop optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=args.eta_min, T_0=args.T0, T_mult=args.T_mult)
    if args.resume:
        model.load_state_dict(torch.load(args.checkpoint_path))
        current_step = int(args.checkpoint_path.split('/')[-1].split('-')[-1])
        print("loaded model: %s" % args.label_checkpoint_path)
    else:
        current_step = 0
    # smooth means average. Every batch has a mean loss value w.r.t. different losses
    smooth_total_loss = 0.0  # total loss
    smooth_AP = 0.0
    smooth_HL = 0.0
    best_AP = 0.0  # best AP for ckpt selection in validation
    best_HL = 0.0  # best HL for ckpt selection in validation

    # training the model
    for one_epoch in range(args.max_epoch):
        if one_epoch:
            scheduler.step()
        print('epoch ' + str(one_epoch + 1) + ' starts!')
        np.random.shuffle(train_idx)  # random shuffle the training indices

        for i in range(int(len(train_idx) / float(args.batch_size)) + 1):
            optimizer.zero_grad()
            start = i * args.batch_size
            end = min(args.batch_size * (i + 1), len(train_idx))

            input_feat=train_feature[train_idx[start:end],:]
            input_label=train_label[train_idx[start:end],:]
            input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
            input_label = deepcopy(input_label).float().to(device)
            label_embed=label_embed.to(device)
            A=A.to(device)
            y_z_mu,y_z_logvar,y_z,x_z_mu,x_z_logvar,x_z,x_zs_mu,x_zs_logvar,xz_y,yz_x,z_y= model(input_feat, input_label,label_embed,A,mode='train')

            total_loss = compute_loss(y_z_mu,y_z_logvar,y_z,x_z_mu,x_z_logvar,x_z,x_zs_mu,x_zs_logvar,xz_y,yz_x,z_y,input_feat,input_label,args.lambda1,args.lambda2,args.lambda3)
            pred_x=xz_y
            total_loss.backward()
            optimizer.step()
            train_metrics = evals.compute_metrics(pred_x.cpu().data.numpy(), input_label.cpu().data.numpy(), 0.5,
                                                  all_metrics=False)
            AP, HL = train_metrics['AP'], train_metrics['HL']

            smooth_total_loss += total_loss
            smooth_AP += AP
            smooth_HL += HL

            current_step += 1
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', lr, current_step)

            if current_step % args.check_freq == 0:  # summarize the current training status and print them out
                total_loss = smooth_total_loss / float(args.check_freq)
                AP = smooth_AP / float(args.check_freq)
                HL = smooth_HL / float(args.check_freq)
                print(
                "step=%d\t, AP=%.6f\t, HL=%.6f\t, loss=%.6f\n" % (current_step, AP, HL,total_loss))
                smooth_total_loss = 0
                smooth_AP = 0
                smooth_HL = 0

            if current_step % int(one_epoch_iter) == 0:  # exam the model on validation set
                print("--------------------------------")
                # exam the model on validation set
                current_loss, val_metrics = valid(val_feature, val_label, model, writer, current_step, args)

                optimizer.zero_grad()

                # select the best checkpoint based on some metric on the validation set
                if (val_metrics['AP']) > best_AP:
                    best_AP = val_metrics['AP']
                    best_HL=val_metrics['HL']
                    best_iter = current_step
                    print('saving model')
                    torch.save(model.state_dict(), model_dir + '/FDML-' + str(current_step))
                    print('have saved model to ', model_dir)
                    print()
                if math.fabs(val_metrics['AP'] - best_AP) < 1e-5:
                        if val_metrics['HL'] < best_HL:
                            best_AP = val_metrics['AP']
                            best_HL = val_metrics['HL']
                            best_iter = current_step
                            print('saving model')
                            torch.save(model.state_dict(), model_dir + '/FDML-' + str(current_step))
                            print('have saved model to ', model_dir)
                            print()

                print("--------------------------------")

def valid(val_feature,val_label, model, summary_writer, current_step, args):
    model.eval()
    print("performing validation...")
    all_total_loss = 0
    all_pred_x = []
    all_label = []
    valid_idx=[i for i in range(len(val_feature))]
    real_batch_size = min(args.batch_size, len(valid_idx))
    for i in range(int((len(valid_idx) - 1) / real_batch_size) + 1):
        start = real_batch_size * i
        end = min(real_batch_size * (i + 1), len(valid_idx))
        input_feat=val_feature[valid_idx[start:end],:]
        input_label=val_label[valid_idx[start:end],:]
        input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
        input_label = deepcopy(input_label).float().to(device)
        with torch.no_grad():
            y_z_mu,y_z_logvar,y_z,x_z_mu,x_z_logvar,x_z,x_zs_mu,x_zs_logvar,xz_y,yz_x,z_y= model(input_feat, input_label, mode='test')
            total_loss = compute_loss(y_z_mu, y_z_logvar, y_z, x_z_mu, x_z_logvar,x_z, x_zs_mu,x_zs_logvar,xz_y, yz_x,z_y, input_feat, input_label,args.lambda1, args.lambda2, args.lambda3)
            pred_x=xz_y
        all_total_loss += total_loss * (end - start)
        all_pred_x.append(pred_x)
        all_label.append(input_label)
    # collect all predictions and ground-truths
    all_pred_x = torch.cat(all_pred_x).detach().cpu().numpy()
    all_label = torch.cat(all_label).detach().cpu().numpy()
    total_loss = all_total_loss / len(valid_idx)


    def show_results(all_indiv_prob):
        best_val_metrics = None
        for threshold in THRESHOLDS:
            val_metrics = evals.compute_metrics(all_indiv_prob, all_label, threshold, all_metrics=True)

            if best_val_metrics == None:
                best_val_metrics = {}
                for metric in METRICS:
                    best_val_metrics[metric] = val_metrics[metric]

        rl,oe,cv,ap = best_val_metrics['RL'], best_val_metrics['OE'],best_val_metrics['CV'],best_val_metrics['AP']

        print("**********************************************")
        print(
            "valid results: RL=%.6f\t,OE=%.6f\t,CV=%.6f\t,AP=%.6f\t,total_loss=%.6f\n" % (rl,oe,cv,ap, total_loss))
        print("**********************************************")

        return rl,oe,cv,ap,best_val_metrics

    rl,oe,cv,ap, best_val_metrics = show_results(all_pred_x)

    summary_writer.add_scalar('valid/total_loss', total_loss, current_step)
    summary_writer.add_scalar('valid/RL', rl, current_step)
    summary_writer.add_scalar('valid/OE', oe, current_step)
    summary_writer.add_scalar('valid/CV', cv, current_step)
    summary_writer.add_scalar('valid/AP', ap, current_step)
    return total_loss, best_val_metrics

