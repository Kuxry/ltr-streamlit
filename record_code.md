
def get_delta_ndcg(batch_ideal_rankings, batch_predict_rankings, device='cpu'):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_ideal_rankings: the standard labels sorted in a descending order
    :param batch_predicted_rankings: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    # ideal discount cumulative gains
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, device=device)

    batch_gains = torch.pow(2.0, batch_predict_rankings) - 1.0

    batch_n_gains = batch_gains / batch_idcgs  # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_predict_rankings.size(1), dtype=torch.float, device=device)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)  # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(
        batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg


class LambdaRank(NeuralRanker):
    '''
    Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
    Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
    '''

    def __init__(self):
        super(LambdaRank, self).__init__(id='LambdaRank')
        self.sigma = 1.0

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        assert 'presort' in kwargs and kwargs['presort'] is True  # aiming for direct usage of ideal ranking

        batch_descending_preds, batch_pred_desc_inds = torch.sort(batch_preds, dim=1,
                                                                  descending=True)  # sort documents according to the predicted relevance
        batch_predict_rankings = torch.gather(batch_std_labels, dim=1,
                                              index=batch_pred_desc_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        batch_std_diffs = torch.unsqueeze(batch_predict_rankings, dim=2) - torch.unsqueeze(batch_predict_rankings,
                                                                                           dim=1)  # standard pairwise differences, i.e., S_{ij}
        batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_std_Sij)

        batch_s_ij = torch.unsqueeze(batch_descending_preds, dim=2) - torch.unsqueeze(batch_descending_preds,
                                                                                      dim=1)  # computing pairwise differences, i.e., s_i - s_j
        # batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)
        batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)

        batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_std_labels,
                                          batch_predict_rankings=batch_predict_rankings, device=self.device)

        # a direct setting of reduction='mean' is meaningless due to breaking the query unit, which also leads to poor performance
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1),
                                             weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='none')

        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss
