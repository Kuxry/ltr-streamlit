
import torch
import torch.nn as nn
import torch.optim as optim

from ndcg import torch_ndcg_at_ks, torch_ndcg_at_k


class NeuralRanker():
    """
    NeuralRanker is a class that represents a general learning-to-rank model.
    Different learning-to-rank models inherit NeuralRanker, but differ in custom_loss_function, which corresponds to a particular loss function.
    """

    def __init__(self, id='AbsRanker', gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device
        self.init()

    def init(self):
        # inner scoring function by using a hard-coded one as an example
        self.sf = nn.Sequential(
            nn.Linear(46, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1), nn.GELU())
        self.optimizer = optim.Adam(self.sf.parameters(), lr=0.0001, weight_decay=0.0001)

    def eval_mode(self):
        '''
        model.eval() is a kind of switch for some specific layers/parts of the model that behave differently
        during training and inference (evaluating) time.
        For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation,
        and .eval() will do it for you. In addition, the common practice for evaluating/validation is using
        torch.no_grad() in pair with model.eval() to turn off gradients computation:
        '''
        self.sf.eval()

    def train_mode(self):
        self.sf.train(mode=True)

    def train(self, train_data, epoch_k=None, **kwargs):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()

        assert 'presort' in kwargs
        presort = kwargs['presort']
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(
                self.device), batch_std_labels.to(self.device)

            batch_loss = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k,
                                       presort=presort)

            epoch_loss += batch_loss.item()

        epoch_loss = epoch_loss / num_queries
        return epoch_loss

    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        '''
        The training operation over a batch of queries.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs: optional arguments
        @return:
        '''
        batch_preds = self.forward(batch_q_doc_vectors)
        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs)

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        The loss function to be customized
        @param batch_preds: [batch, ranking_size] each row represents the predicted relevance values for documents associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs:
        @return:
        '''
        pass

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()

        _batch_preds = self.sf(batch_q_doc_vectors)
        batch_preds = _batch_preds.view(-1, num_docs)  # [batch_size x num_docs, 1] -> [batch_size, num_docs]
        return batch_preds

    def predict(self, batch_q_doc_vectors):
        '''
        The relevance prediction.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_preds = self.forward(batch_q_doc_vectors)
        return batch_preds

    def ndcg_at_k(self, test_data=None, k=10, presort=False, device='cpu'):
        '''
        Compute nDCG@k with the given data
        An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_k = torch.zeros(1)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)

            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_k = torch_ndcg_at_k(batch_predict_rankings=batch_predict_rankings,
                                              batch_ideal_rankings=batch_ideal_rankings,
                                              k=k, device=device)

            sum_ndcg_at_k += torch.sum(batch_ndcg_at_k)  # due to batch processing

        avg_ndcg_at_k = sum_ndcg_at_k / num_queries
        return avg_ndcg_at_k

    def ndcg_at_ks(self, test_data=None, ks=[1, 5, 10], presort=False, device='cpu'):
        '''
        Compute nDCG with multiple cutoff values with the given data
        An underlying assumption is that there is at least one relevant document, or ZeroDivisionError appears.
        '''
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_ndcg_at_ks = torch.zeros(len(ks))
        for batch_ids, batch_q_doc_vectors, batch_std_labels in test_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            if self.gpu: batch_preds = batch_preds.cpu()

            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)
            if presort:
                batch_ideal_rankings = batch_std_labels
            else:
                batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

            batch_ndcg_at_ks = torch_ndcg_at_ks(batch_predict_rankings=batch_predict_rankings,
                                                batch_ideal_rankings=batch_ideal_rankings,
                                                ks=ks, device=device)
            sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.sum(batch_ndcg_at_ks, dim=0))
            num_queries += len(batch_ids)

        avg_ndcg_at_ks = sum_ndcg_at_ks / num_queries
        return avg_ndcg_at_ks

