
import torch
import numpy as np
#
# """ processing on letor datasets """

def torch_dcg_at_k(batch_rankings, cutoff=None, device='cpu'):
    # '''
    # ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    # :param batch_rankings: [batch_size, ranking_size] rankings of labels (either standard or predicted by a system)
    # :param cutoff: the cutoff position
    # :return: [batch_size, 1] cumulative gains for each rank position
    # '''
    if cutoff is None:  # using whole list
        cutoff = batch_rankings.size(1)

    batch_numerators = torch.pow(2.0, batch_rankings[:, 0:cutoff]) - 1.0

    # no expanding should also be OK due to the default broadcasting
    batch_discounts = torch.log2(
        torch.arange(cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators / batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k


def torch_dcg_at_ks(batch_rankings, max_cutoff, device='cpu'):
    # '''
    # :param batch_rankings: [batch_size, ranking_size] rankings of labels (either standard or predicted by a system)
    # :param max_cutoff: the maximum cutoff value
    # :return: [batch_size, max_cutoff] cumulative gains for each ranlok position
    # '''
    batch_numerators = torch.pow(2.0, batch_rankings[:, 0:max_cutoff]) - 1.0

    batch_discounts = torch.log2(
        torch.arange(max_cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_ks = torch.cumsum(batch_numerators / batch_discounts, dim=1)  # dcg w.r.t. each position
    return batch_dcg_at_ks


def torch_ndcg_at_k(batch_predict_rankings, batch_ideal_rankings, k=None, device='cpu'):
    batch_sys_dcg_at_k = torch_dcg_at_k(batch_predict_rankings, cutoff=k,
                                        device=device)  # only using the cumulative gain at the final rank position
    batch_ideal_dcg_at_k = torch_dcg_at_k(batch_ideal_rankings, cutoff=k, device=device)
    batch_ndcg_at_k = batch_sys_dcg_at_k / batch_ideal_dcg_at_k
    return batch_ndcg_at_k


def torch_ndcg_at_ks(batch_predict_rankings, batch_ideal_rankings, ks=None, device='cpu'):
    valid_max_cutoff = batch_predict_rankings.size(1)
    used_ks = [k for k in ks if k <= valid_max_cutoff] if valid_max_cutoff < max(ks) else ks

    inds = torch.from_numpy(np.asarray(used_ks) - 1).type(torch.long)
    batch_sys_dcgs = torch_dcg_at_ks(batch_predict_rankings, max_cutoff=max(used_ks), device=device)
    batch_sys_dcg_at_ks = batch_sys_dcgs[:, inds]  # get cumulative gains at specified rank positions
    batch_ideal_dcgs = torch_dcg_at_ks(batch_ideal_rankings, max_cutoff=max(used_ks), device=device)
    batch_ideal_dcg_at_ks = batch_ideal_dcgs[:, inds]

    batch_ndcg_at_ks = batch_sys_dcg_at_ks / batch_ideal_dcg_at_ks

    if valid_max_cutoff < max(ks):
        padded_ndcg_at_ks = torch.zeros(batch_predict_rankings.size(0), len(ks))
        padded_ndcg_at_ks[:, 0:len(used_ks)] = batch_ndcg_at_ks
        return padded_ndcg_at_ks
    else:
        return batch_ndcg_at_ks

