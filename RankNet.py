
import torch
import torch.nn.functional as F
from NeuralRank import NeuralRanker

class RankNet(NeuralRanker):

    def __init__(self):
        super(RankNet, self).__init__(id='RankNet')
        self.sigma = 1.0

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):#参数：打分函数的输出，对应的标签

        batch_pre_sij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        batch_pre_pij = torch.sigmoid(self.sigma * batch_pre_sij)


        batch_std_sij= torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
        batch_std_one =  torch.clamp(batch_std_sij, min=-1.0, max=1.0)
        batch_std_pij = 0.5 * (1.0 + batch_std_one)

        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_pre_pij, diagonal=1), target=torch.triu(batch_std_pij, diagonal=1), reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss