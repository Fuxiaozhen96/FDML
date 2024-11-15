import os
import torch
import numpy as np
from sklearn.preprocessing import normalize

def build_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + "/" + path_seg
        else:
            cur_path = path_seg
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

def get_label(data, order, feature_dim):
    output = []
    for i in order:
        output.append(data[i,feature_dim:])
    output = np.array(output, dtype="int")
    return output

def get_feat(data, order, feature_dim):
    output = []
    for i in order:
        output.append(data[i,0:feature_dim])
    output = np.array(output, dtype="float32")
    return output

def init_label_embed(features,labels):
    features=torch.from_numpy(features)
    Y_avg = torch.from_numpy(labels).float()
    label_embedding=torch.matmul(Y_avg.T,features)
    embedding_norm = torch.from_numpy(normalize(label_embedding, axis=1, norm="l2")).float()
    label_embedding=torch.div(label_embedding,embedding_norm+1e-6)
    return label_embedding

def get_poex_naex(features,label,k):
    num_example = np.shape(features)[0]
    num_feature = np.shape(features)[1]
    num_label = np.shape(label)[1]
    positive = np.zeros((num_example, num_feature),dtype="float32")
    positive_label = np.zeros((num_example, num_label),dtype="int")
    negative = np.zeros((num_example, num_feature),dtype="float32")
    negative_label = np.zeros((num_example, num_label),dtype="int")
    num_poex = np.sum(label, axis=0)
    for i in range(num_example):
        label_add = label[i] * label
        label_or = label[i] + label
        label_or[label_or >= 1] = 1
        sum_add = np.sum(label_add / num_poex.transpose(), 1)
        sum_or = np.sum(label_or / num_poex.transpose(), 1)
        score = sum_add / sum_or
        lindex = np.argsort(-score)
        x_i = features[i, :]
        l_i = label[i, :]
        x_k = features[lindex[1:k + 1], :]
        l_k = label[lindex[1:k + 1], :]
        f_kscore = np.linalg.norm(x_i - x_k, ord=2, axis=1)
        f_kindex = np.argsort(-f_kscore)
        positive[i] = x_k[f_kindex[0], :]
        positive_label[i] = l_k[f_kindex[0], :]
        temp_label = l_i + positive_label[i]
        temp_label[temp_label > 1] = 1
        f_allscore = np.linalg.norm(x_i - features, ord=2, axis=1)
        f_allindex = np.argsort(f_allscore)
        for n_index in f_allindex:
            if np.sum(label[n_index, :] * temp_label) == 0:
                negative[i] = features[n_index, :]
                negative_label[i] = label[n_index, :]
                break
    return positive,positive_label,negative,negative_label

def log_normal(x, m, v):
    log_prob = torch.sum((-0.5 * (torch.log(v) + (x-m).pow(2) / (v+1e-6))))
    return log_prob

def log_normal_mixture(z, m, v, mask=None):
    m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
    v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
    batch, mix, dim = m.size()
    z = z.view(batch, 1, dim).expand(batch, mix, dim)
    indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask)*(-1e6)*(1.-mask)
    log_prob = log_mean_exp(indiv_log_prob, mask)
    return log_prob

def log_mean_exp(x, mask):
    return log_sum_exp(x, mask) - torch.log(mask.sum(1))

def log_sum_exp(x, mask):
    max_x = torch.max(x, 1)[0]
    new_x = x - max_x.unsqueeze(1).expand_as(x)
    return max_x + (new_x.exp().sum(1)).log()

THRESHOLDS = [0.5]



def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2
    return stats

def gen_A(label):
    num_class=np.shape(label)[1]
    label=torch.from_numpy(label)
    coocurence = torch.matmul(label.T, label)
    num_exam = torch.sum(label, dim=0).resize(num_class, 1)
    num_exam = num_exam.repeat(1, num_class) + 1e-6
    _adj = torch.div(coocurence, num_exam)
    _adj[_adj < 0.1] = 0
    _adj[_adj >= 0.1] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + torch.eye(num_class)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj