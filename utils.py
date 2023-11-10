import torch
from torch import nn
import numpy as np
from torch.nn.functional import normalize


def avg_both(mrrs, hits):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    # print("mrrs:", mrrs)
    # print("hits:", hits)
    return {'MRR': m, 'hits@[1,3,10]': h}


def temporal_regularizer(factor, weight, M):
    ##MSE
    ddiff = factor[1:] - factor[:-1]
    diff = ddiff**2
    time_diff = torch.sum(diff) / (factor.shape[0] - 1)
    return weight * time_diff

def temporal_regularizer1(factor, weight,M):
    #Gaussian
    sigma = 0.01
    ddiff = factor[1:] - factor[:-1]
    diff = - ddiff**2 / (2 * sigma * sigma)
    time_diff = weight * torch.exp(torch.sum(diff)/ (factor.shape[0] - 1))
    return - weight * time_diff

def temporal_regularizer2(factor, weight, M):
    #cosine
    factor = normalize(factor)
    sim = torch.mm(factor, factor.T)
    N = factor.shape[0]
    positive1 = torch.sum(torch.diag(sim, 1))/(N - 1)
    positive2 = torch.sum(torch.diag(sim, -1))/(N - 1)
    time_diff = positive1 + positive2
    return -weight * time_diff

def temporal_regularizer3(factor, weight, M):
    #contrastive learning
    criterion_node = nn.CrossEntropyLoss(reduction="sum")
    factor = normalize(factor)
    sim  = torch.mm(factor, factor.T)
    N = factor.shape[0]
    labels_node = torch.zeros(N - 1).to(factor.device).long()

    positive1 = torch.diag(sim, 1)
    positive2 = torch.diag(sim, -1)
    positive_samples = positive1.reshape(N-1, 1)
################
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N -1):
        mask[i, 1 + i] = 0
    for i in range(N - 1):
        mask[i+1, i] = 0
    mask = mask.bool()
    negative_samples = sim[mask].reshape(N-1, N-2)
    logits_node = torch.cat((positive_samples, negative_samples), dim=1)
    loss_t = criterion_node(logits_node, labels_node)
    loss_t /= N
    return -weight * loss_t

def temporal_regularizer4(factor, weight, M):
    #use weight to normalize time embedding
    N = factor.shape[0]
    factor = normalize(factor)
    sim = torch.mm(factor, factor.T)
    time_diff = torch.sum((sim - M)**2)
    time_diff /= N
    return weight * time_diff


def emb_regularizer(factors, weight):
    norm = 0
    for f in factors:
        norm += weight * torch.abs(f) ** 2
    return norm/3

def creat_M(args, sizes):
    N = sizes[3]
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            i = i
            j = j
            M[i][j] = np.exp(-np.abs(i - j) / args.sigma)
    M = torch.tensor(M)
    return M