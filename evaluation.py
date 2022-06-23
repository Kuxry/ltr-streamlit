
import streamlit as st
import pandas as pd
import torch
import altair as alt
import datetime
import numpy as np

from datasets_load import LTRDataset, LETORSampler
from pre_datasets import SPLIT_TYPE

from LambdaRank import *
from RankMse import*
from RankNet import*


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

