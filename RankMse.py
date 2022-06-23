

import torch
import torch.nn.functional as F
from NeuralRank import NeuralRanker


def rankMSE_loss_function(rel_preds=None, std_labes=None):
    tensor_batch_loss=F.mse_loss(rel_preds,std_labes,reduction='none')
    batch_loss= torch.mean(torch.sum(tensor_batch_loss))
    return batch_loss

class RankMSE(NeuralRanker):
    def __int__(self):
        super(RankMSE, self).__int__(id='RankMSE')

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        batch_loss = rankMSE_loss_function(batch_preds,batch_std_labels)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


