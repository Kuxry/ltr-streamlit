
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import altair as alt
import datetime
import numpy as np

from datasets_load import LTRDataset, LETORSampler
from ndcg import torch_ndcg_at_ks, torch_ndcg_at_k, torch_dcg_at_k
from pre_datasets import SPLIT_TYPE


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


def rankMSE_loss_function(relevance_preds=None, std_labels=None):
    '''
    Ranking loss based on mean square error to do adjust output scale w.r.t. output layer activation function
    @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
    @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
    @return:
    '''
    _batch_loss = F.mse_loss(relevance_preds, std_labels, reduction='none')
    batch_loss = torch.mean(torch.sum(_batch_loss, dim=1))
    return batch_loss


class RankMSE(NeuralRanker):
    def __init__(self):
        super(RankMSE, self).__init__(id='RankMSE')

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''
        # print('batch_preds', batch_preds.size())
        # print(batch_preds)
        batch_loss = rankMSE_loss_function(batch_preds, batch_std_labels)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


class RankNet(NeuralRanker):
    '''
    Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005.
    Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.
    '''

    def __init__(self):
        super(RankNet, self).__init__(id='RankNet')
        self.sigma = 1.0

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds,
                                                                           dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        # batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)
        batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)

        batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels,
                                                                                     dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss


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


def load_multiple_data(data_id, dir_data, fold_k, batch_size=100):
    # """
    # Load the dataset correspondingly.
    # :param eval_dict:
    # :param data_dict:
    # :param fold_k:
    # :param model_para_dict:
    # :return:
    # """
    fold_k_dir = dir_data + 'Fold' + str(fold_k) + '/'
    file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

    _train_data = LTRDataset(data_id=data_id, file=file_train, split_type=SPLIT_TYPE.Train, batch_size=batch_size)
    train_letor_sampler = LETORSampler(data_source=_train_data, rough_batch_size=batch_size)
    train_data = torch.utils.data.DataLoader(_train_data, batch_sampler=train_letor_sampler, num_workers=0)

    _test_data = LTRDataset(data_id=data_id, file=file_test, split_type=SPLIT_TYPE.Test, batch_size=batch_size)
    test_letor_sampler = LETORSampler(data_source=_test_data, rough_batch_size=batch_size)
    test_data = torch.utils.data.DataLoader(_test_data, batch_sampler=test_letor_sampler, num_workers=0)

    return train_data, test_data


def metric_results_to_string(list_scores=None, list_cutoffs=None, split_str=', ', metric='nDCG'):
    # """
    # Convert metric results to a string representation
    # :param list_scores:
    # :param list_cutoffs:
    # :param split_str:
    # :return:
    # """
    list_str = []
    for i in range(len(list_scores)):
        list_str.append(metric + '@{}:{:.4f}'.format(list_cutoffs[i], list_scores[i]))
    return split_str.join(list_str)


def evaluation(data_id=None, dir_data=None, model_id=None, batch_size=100):
    # """
    # Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
    # :param data_dict:       settings w.r.t. data
    # :param eval_dict:       settings w.r.t. evaluation
    # :param sf_para_dict:    settings w.r.t. scoring function
    # :param model_para_dict: settings w.r.t. the ltr_adhoc model
    # :return:
    # """
    fold_num = 5
    cutoffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cutoffs_line = [1, 3, 5, 10]
    epochs = 2
    ranker = globals()[model_id]()

    time_begin = datetime.datetime.now()  # timing
    l2r_cv_avg_scores = np.zeros(len(cutoffs))  # fold average

    for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
        ranker.init()  # initialize or reset with the same random initialization

        train_data, test_data = load_multiple_data(data_id=data_id, dir_data=dir_data, fold_k=fold_k)
        # test_data = None

        for epoch_k in range(1, epochs + 1):
            torch_fold_k_epoch_k_loss = ranker.train(train_data=train_data, epoch_k=epoch_k, presort=True)

        torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=test_data, ks=cutoffs, device='cpu', presort=True)
        fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()  # .data是读取Variable中的tensor .numpy()把tensor变成numpy 从格式data的张量  到numpy的数组格式

        # 折线图
        source = pd.DataFrame(fold_ndcg_ks,
                              columns=[model_id + ' Fold-' + str(fold_k)],
                              index=pd.RangeIndex(start=1, stop=11, name='x'))
        source = source.reset_index().melt('x', var_name='Category', value_name='y')

        line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
            alt.X('x', title='The Number Of DCG'),
            alt.Y('y', title='NDCG value'),
            color='Category:N'
        ).properties(
            title='NDCG Score Trends'
        )

        performance_list = [model_id + ' Fold-' + str(fold_k)]  # fold-wise performance
        for i, co in enumerate(cutoffs_line):
            performance_list.append('nDCG@{}:{:.4f}'.format(co, fold_ndcg_ks[i]))
            # st.write(format(fold_ndcg_ks[i])
        performance_str = '\t'.join(performance_list)
        # st.line_chart(chart_data)
        # st.altair_chart(c, use_container_width=True)
        st.altair_chart(line_chart)
        st.write(performance_str)
        # st.write('\t', performance_str)

        l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks)  # sum for later cv-performance

    time_end = datetime.datetime.now()  # overall timing
    elapsed_time_str = str(time_end - time_begin)
    st.subheader("The elapsed time of five folds")
    # st.write('Elapsed time:\t', elapsed_time_str + "\n\n")
    st.metric(label=" ", value=elapsed_time_str)

    l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
    eval_prefix = str(fold_num) + '-fold average scores:'
    # st.write(model_id, eval_prefix, metric_results_to_string(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))  # print either cv or average performance

    return l2r_cv_avg_scores


# Add a selectbox to the sidebar:


st.set_page_config(layout="wide", page_title="Learning To Rank")

dataset, loss_function, evaluation1 = st.columns(3)
with dataset:
    st.sidebar.header("Which dataset do you want use?")
    dataset_mode = st.sidebar.selectbox("Choose the mode",
                                        ["Microsoft", "Yahoo"])

with loss_function:
    st.sidebar.header("Which loss function do you want use?")
    lf_mode = st.sidebar.selectbox("Choose the mode",
                                   ["RankMSE", "RankNet", "LambdaRank"])

with evaluation1:
    st.sidebar.header("Which evaluation do you want use?")
    eva_mode = st.sidebar.selectbox("Choose the mode",
                                    ["NDCG", "Other"])

st.sidebar.title("Batch size")
batchsize = st.sidebar.slider('', 1, 100)
# st.write(x, 'squared is', x * x)
st.sidebar.write(batchsize)

# 正文
st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Learning To Rank👋</h1>", unsafe_allow_html=True)
# 可扩展段落
st.write("")
st.markdown(
    """

    **Learning to Rank** is a machine Learning model. It uses machine learning methods, 
    we can take the [output as feature] of each existing ranking model, and then train a new model, and automatically learn the parameters of the new model, 
    so it is very convenient to combine multiple existing ranking model to generate a new ranking model.

    **👈 Select a way you want to try from the dropdown on the left** to see some results
    of what Learning To Rank can do!

    ### What is the dataset?

    """
)
with st.expander("Mircosoft"):
    st.markdown("""
        **LETOR(Learning to Rank for Information Retrieval)** is a package of benchmark data sets for research on Learning To Rank, 
        which contains standard features, relevance judgments, data partitioning, evaluation tools, and several baselines. 

        There are about 1700 queries in MQ2007 with labeled documents and about 800 queries in MQ2008 with labeled documents.

        ### Datasets
        The 5-fold cross validation strategy is adopted and the 5-fold partitions are included in the package. 
        In each fold, there are three subsets for learning: training set, validation set and testing set.
        ##### Descriptions
        Each row is a query-document pair. The first column is **relevance label** of this pair, 
        the second column is **query id**, the following columns are **features**, and the end of the row is **comment** about the pair, including id of the document.
        **The larger the relevance label, the more relevant the query-document pair.**
        A query-document pair is represented by a 46-dimensional feature vector. 

        Here are several example rows from MQ2007 dataset:

        ----
        > 2 qid:10032 1:0.056537 2:0.000000 3:0.666667 4:1.000000 5:0.067138 … 45:0.000000 46:0.076923 #docid = GX029-35-5894638 inc = 0.0119881192468859 prob = 0.139842

        > 0 qid:10032 1:0.279152 2:0.000000 3:0.000000 4:0.000000 5:0.279152 … 45:0.250000 46:1.000000 #docid = GX030-77-6315042 inc = 1 prob = 0.341364

        > 0 qid:10032 1:0.130742 2:0.000000 3:0.333333 4:0.000000 5:0.134276 … 45:0.750000 46:1.000000 #docid = GX140-98-13566007 inc = 1 prob = 0.0701303

        > 1 qid:10032 1:0.593640 2:1.000000 3:0.000000 4:0.000000 5:0.600707 … 45:0.500000 46:0.000000 #docid = GX256-43-0740276 inc = 0.0136292023050293 prob = 0.400738



        """)

with st.expander("Yahoo"):
    st.markdown("""

        """)

st.markdown(
    """

    - Mircosoft dataset
     [LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)
    - Yahoo dataset
     [Yahoo](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
)

st.markdown(
    """
    ### What you can get?
    - The NDCG score of each fold
    - The elapsed time of five folds
    - The NDCG average score of five folds

    - Displays changes in NDCG Average Score

    """
)

st.title("OUTPUT")
Result, NDCG_score = st.columns(2)
with Result:
    # split_type=SPLIT_TYPE.Train
    st.subheader("Result")
    # train_data = load_data(data_id=data_id, file=file_train, split_type=split_type, batch_size=batch_size)
    # # st.write(train_data.batch_size)
    #
    # for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:
    #     st.write(batch_q_doc_vectors.shape)
    data_id = 'MQ2008_Super'
    dir_data = '/home/uba/data/MQ2008/'
    average = evaluation(data_id=data_id, dir_data=dir_data, model_id=lf_mode, batch_size=batchsize)

st.subheader(" ")
with NDCG_score:
    st.subheader("NDCG Score")
    pb = st.progress(0)
    status_txt = st.empty()

    for i in range(100):
        pb.progress(i)
        new_rows = average
        status_txt.text(
            "The 5-fold average scores : %s" % new_rows
        )

    # 折线图
    source_average = pd.DataFrame(average,
                                  columns=['NDCG Average Scores'], index=pd.RangeIndex(start=1, stop=11, name='x'))
    source_average = source_average.reset_index().melt('x', var_name='Average Scores', value_name='y')

    line_chart = alt.Chart(source_average).mark_line(interpolate='basis').encode(
        alt.X('x', title='The Number Of DCG'),
        alt.Y('y', title='NDCG value'),
        color='Average Scores:N'
    ).properties(
        title='NDCG Average Score Trends'
    )
    st.altair_chart(line_chart)

    st.subheader("Displays changes in NDCG Average Score")
    for i in range(10):
        st.metric(label="nDCG average scores", value=average[i], delta=average[i] - average[i - 1])
