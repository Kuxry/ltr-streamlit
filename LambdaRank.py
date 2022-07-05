
import torch
import torch.nn.functional as F
from NeuralRank import NeuralRanker
from ndcg import torch_dcg_at_k


def get_delta_ndcg(batch_ideal_rankings, batch_predict_rankings, device='cpu'):
    batch_idcgs= torch_dcg_at_k(batch_rankings=batch_ideal_rankings,device=device)
    batch_gains= torch.pow(2.0 , batch_predict_rankings) - 1.0

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

    def __int__(self):
        super(LambdaRank, self).__int__(id='LambdaRank')
        self.sigma = 1.0

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):

        #sorting
        batch_desc_preds, batch_pred_desc_inds= torch.sort(batch_preds, dim=1,descending=True)
        #batch_desc_stds = torch.gather(batch_std_labels,dim=1,index=batch_pred_desc_inds)
        batch_desc_stds, batch_desc_stds_inds= torch.sort(batch_std_labels, dim=1,descending=True)

        #pre
        batch_pre_sij= torch.unsqueeze(batch_desc_preds,dim=2) - torch.unsqueeze(batch_desc_preds,dim=1)
        batch_pre_pij= torch.sigmoid(self.sigma * batch_pre_sij)

        #std
        batch_std_sij= torch.unsqueeze(batch_desc_stds,dim=2) - torch.unsqueeze(batch_desc_stds,dim=1)
        batch_std_one= torch.clamp(batch_std_sij,min=-1.0, max= 1.0)
        batch_std_pij = 0.5 * (1.0 + batch_std_one)

        #delta
        batch_delta_ndcg= get_delta_ndcg(batch_ideal_rankings=batch_std_labels,
                                          batch_predict_rankings=batch_desc_stds, device=self.device)


        _batch_loss= F.binary_cross_entropy(input=torch.triu(batch_pre_pij,diagonal=1),
                                            target=torch.triu(batch_std_pij,diagonal=1),
                                            weight=torch.triu(batch_delta_ndcg,diagonal=1),reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss





