class RankNet(NeuralRanker):
    '''
    Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
    Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
    '''
    def __init__(self):
        super(RankNet, self).__init__(id='RankNet')
        self.sigma = 1.0

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):#参数：打分函数的输出，对应的标签
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        batch>1 表示多个查询文本
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        '''
       [1,3]

        [1,3]-[1,3,1]
        [1,3]-[1.1.3]

        '''
        #计算某个文档排在某个文档前的概率值
        #以tensor为单位，进行批处理
        #dim指在某个地方多一个维度的值
        #batch批处理 同时进行一个batch的si-sj的运算，broadcasting中的功能
        batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        #batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)
        batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)

        #一个文档排在另一个文档的概率值。得到标准的概率值，batch批处理 同时进行一个batch的标准预测的运算
        batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}，2的时候变为1，为-2的时候变为-1。为后面方便计算

        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)
        #损失函数
        #去掉方正的对角线上的值做 loss
        #diagonal=1(去掉对角线的部分)  torch.triu？去掉下部分以及对角线的值，只剩上部分的值
        #F.binary_cross_entropy：Function that measures the Binary Cross Entropy between the target and input probabilities.
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss