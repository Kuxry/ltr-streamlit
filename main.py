import time

import streamlit as st

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import altair as alt
import datetime
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


#
# """ processing on letor datasets """
import os
import pickle
from pathlib import Path

import torch.utils.data as data

## due to the restriction of 4GB ##
max_bytes = 2 ** 31 - 1


def pickle_save(target, file):
    bytes_out = pickle.dumps(target, protocol=4)
    with open(file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load(file):
    file_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        bytes_in = bytearray(0)
        for _ in range(0, file_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data


from enum import Enum, unique, auto


@unique
class MASK_TYPE(Enum):
    # """ Supported ways of masking labels """
    rand_mask_all = auto()
    rand_mask_rele = auto()


@unique
class LABEL_TYPE(Enum):
    # """ The types of labels of supported datasets """
    MultiLabel = auto()
    Permutation = auto()


@unique
class SPLIT_TYPE(Enum):
    # """ The split-part of a dataset """
    Train = auto()
    Test = auto()
    Validation = auto()


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALER_ID = ['MinMaxScaler', 'RobustScaler', 'StandardScaler', "SLog1P"]
SCALER_LEVEL = ['QUERY', 'DATASET']
MSLETOR_SEMI = ['MQ2007_Semi', 'MQ2008_Semi']
MSLETOR_LIST = ['MQ2007_List', 'MQ2008_List']
MSLETOR_SUPER = ['MQ2007_Super', 'MQ2008_Super']
MSLETOR = ['MQ2007_Super', 'MQ2008_Super', 'MQ2007_Semi', 'MQ2008_Semi', 'MQ2007_List', 'MQ2008_List']

# '''
# The dataset used in the IRGAN paper, which is a revised version of MQ2008_Semi by adding some document vectors per query
# in order to mimic unlabeled documents. Unfortunately, the details on how to generate these personally added documents
# are not described.
# '''
IRGAN_MQ2008_SEMI = ['IRGAN_MQ2008_Semi']

MSLRWEB = ['MSLRWEB10K', 'MSLRWEB30K']

YAHOO_LTR = ['Set1', 'Set2']
YAHOO_LTR_5Fold = ['5FoldSet1', '5FoldSet2']

ISTELLA_LTR = ['Istella_S', 'Istella', 'Istella_X']
ISTELLA_MAX = 1000000  # As ISTELLA contain extremely large features, e.g., 1.79769313486e+308, we replace features of this kind with a constant 1000000

GLTR_LIBSVM = ['LTR_LibSVM', 'LTR_LibSVM_K']
GLTR_LETOR = ['LETOR', 'LETOR_K']


def get_scaler(scaler_id):
    # """ Initialize the scaler-object correspondingly """
    assert scaler_id in SCALER_ID
    if scaler_id == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_id == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_id == 'StandardScaler':
        scaler = StandardScaler()

    return scaler


def get_data_meta(data_id=None):
    # """ Get the meta-information corresponding to the specified dataset """
    if data_id in MSLRWEB:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 136
        has_comment = False
        fold_num = 5

    elif data_id in MSLETOR_SUPER:
        max_rele_level = 2
        label_type = LABEL_TYPE.MultiLabel
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in MSLETOR_SEMI:
        max_rele_level = 2
        label_type = LABEL_TYPE.MultiLabel
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in MSLETOR_LIST:
        max_rele_level = None
        label_type = LABEL_TYPE.Permutation
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in YAHOO_LTR:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 700  # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 1

    elif data_id in YAHOO_LTR_5Fold:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 700  # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 5

    elif data_id in ISTELLA_LTR:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 220  # libsvm format, rather than uniform number
        fold_num = 1
        if data_id in ['Istella_S', 'Istella']:
            has_comment = False
        else:
            has_comment = True
    else:
        raise NotImplementedError

    data_meta = dict(num_features=num_features, has_comment=has_comment, label_type=label_type,
                     max_rele_level=max_rele_level, fold_num=fold_num)
    return data_meta


def np_arg_shuffle_ties(vec, descending=True):
    # ''' the same as np_shuffle_ties, but return the corresponding indice '''
    if len(vec.shape) > 1:
        raise NotImplementedError
    else:
        length = vec.shape[0]
        perm = np.random.permutation(length)
        if descending:
            sorted_shuffled_vec_inds = np.argsort(-vec[perm])
        else:
            sorted_shuffled_vec_inds = np.argsort(vec[perm])

        shuffle_ties_inds = perm[sorted_shuffled_vec_inds]
        return shuffle_ties_inds


def _parse_docid(comment):
    parts = comment.strip().split()
    return parts[2]


def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]


def get_buffer_file_name(data_id, file, data_dict, presort=None):
    # """ Generate the file name """
    min_rele = data_dict['min_rele']
    if min_rele is not None and min_rele > 0:
        fi_suffix = '_'.join(['MiR', str(min_rele)])
    else:
        fi_suffix = ''

    min_docs = data_dict['min_docs']
    if min_docs is not None and min_docs > 0:
        if len(fi_suffix) > 0:
            fi_suffix = '_'.join([fi_suffix, 'MiD', str(min_docs)])
        else:
            fi_suffix = '_'.join(['MiD', str(min_docs)])

    res_suffix = ''
    if data_dict['binary_rele']:
        res_suffix += '_B'
    if data_dict['unknown_as_zero']:
        res_suffix += '_UO'

    pq_suffix = '_'.join([fi_suffix, 'PerQ']) if len(fi_suffix) > 0 else 'PerQ'

    assert presort is not None
    if presort: pq_suffix = '_'.join([pq_suffix, 'PreSort'])

    # plus scaling
    scale_data = data_dict['scale_data']
    scaler_id = data_dict['scaler_id'] if 'scaler_id' in data_dict else None
    scaler_level = data_dict['scaler_level'] if 'scaler_level' in data_dict else None
    if scale_data:
        assert scaler_id is not None and scaler_id in SCALER_ID and scaler_level in SCALER_LEVEL
        if 'DATASET' == scaler_level:
            pq_suffix = '_'.join([pq_suffix, 'DS', scaler_id])
        else:
            pq_suffix = '_'.join([pq_suffix, 'QS', scaler_id])

    if data_id in YAHOO_LTR:
        perquery_file = file[:file.find('.txt')].replace(data_id.lower() + '.',
                                                         'Buffered' + data_id + '/') + '_' + pq_suffix + res_suffix + '.np'
    elif data_id in ISTELLA_LTR:
        perquery_file = file[:file.find('.txt')].replace(data_id,
                                                         'Buffered_' + data_id) + '_' + pq_suffix + res_suffix + '.np'
    else:
        perquery_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' + pq_suffix + res_suffix + '.np'

    return perquery_file


def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0, has_comment=False):
    # """
    # Transforms an iterator of lines to an iterator of LETOR rows. Each row is represented by a (x, y, qid, comment) tuple.
    # Parameters
    # ----------
    # lines : iterable of lines Lines to parse.
    # has_targets : bool, optional, i.e., the relevance label
    #     Whether the file contains targets. If True, will expect the first token  every line to be a real representing
    #     the sample's target (i.e. score). If False, will use -1 as a placeholder for all targets.
    # one_indexed : bool, optional, i.e., whether the index of the first feature is 1
    #     Whether feature ids are one-indexed. If True, will subtract 1 from each feature id.
    # missing : float, optional
    #     Placeholder to use if a feature value is not provided for a sample.
    # Yields
    # ------
    # x : array of floats Feature vector of the sample.
    # y : float Target value (score) of the sample, or -1 if no target was parsed.
    # qid : object Query id of the sample. This is currently guaranteed to be a string.
    # comment : str Comment accompanying the sample.
    # """
    for line in lines:
        # print(line)
        if has_comment:
            data, _, comment = line.rstrip().partition('#')
            toks = data.split()
        else:
            toks = line.rstrip().split()

        num_features = 0
        feature_vec = np.repeat(missing, 8)
        std_score = -1.0
        if has_targets:
            std_score = float(toks[0])
            toks = toks[1:]

        qid = _parse_qid_tok(toks[0])

        for tok in toks[1:]:
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1

            assert fid >= 0
            while len(feature_vec) <= fid:
                orig = len(feature_vec)
                feature_vec.resize(len(feature_vec) * 2)
                feature_vec[orig:orig * 2] = missing

            feature_vec[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        feature_vec.resize(num_features)

        if has_comment:
            yield (feature_vec, std_score, qid, comment)
        else:
            yield (feature_vec, std_score, qid)


def parse_letor(source, has_targets=True, one_indexed=True, missing=0.0, has_comment=False):
    # """
    # Parses a LETOR dataset from `source`.
    # Parameters
    # ----------
    # source : string or iterable of lines String, file, or other file-like object to parse.
    # has_targets : bool, optional
    # one_indexed : bool, optional
    # missing : float, optional
    # Returns
    # -------
    # X : array of arrays of floats Feature matrix (see `iter_lines`).
    # y : array of floats Target vector (see `iter_lines`).
    # qids : array of objects Query id vector (see `iter_lines`).
    # comments : array of strs Comment vector (see `iter_lines`).
    # """
    max_width = 0
    feature_vecs, std_scores, qids = [], [], []
    if has_comment:
        comments = []

    it = iter_lines(source, has_targets=has_targets, one_indexed=one_indexed, missing=missing, has_comment=has_comment)
    if has_comment:
        for f_vec, s, qid, comment in it:
            feature_vecs.append(f_vec)
            std_scores.append(s)
            qids.append(qid)
            comments.append(comment)
            max_width = max(max_width, len(f_vec))
    else:
        for f_vec, s, qid in it:
            feature_vecs.append(f_vec)
            std_scores.append(s)
            qids.append(qid)
            max_width = max(max_width, len(f_vec))

    assert max_width > 0
    all_features_mat = np.ndarray((len(feature_vecs), max_width), dtype=np.float64)
    all_features_mat.fill(missing)
    for i, x in enumerate(feature_vecs):
        all_features_mat[i, :len(x)] = x

    all_labels_vec = np.array(std_scores)

    if has_comment:
        docids = [_parse_docid(comment) for comment in comments]
        # features, std_scores, qids, docids
        return all_features_mat, all_labels_vec, qids, docids
    else:
        # features, std_scores, qids
        return all_features_mat, all_labels_vec, qids


def clip_query_data(qid, list_docids=None, feature_mat=None, std_label_vec=None, binary_rele=False,
                    unknown_as_zero=False, clip_query=None, min_docs=None, min_rele=1, presort=None):
    # """ Clip the data associated with the same query if required """
    if binary_rele: std_label_vec = np.clip(std_label_vec, a_min=-10, a_max=1)  # to binary labels
    if unknown_as_zero: std_label_vec = np.clip(std_label_vec, a_min=0, a_max=10)  # convert unknown as zero

    if clip_query:
        if feature_mat.shape[0] < min_docs:  # skip queries with documents that are fewer the pre-specified min_docs
            return None
        if (std_label_vec > 0).sum() < min_rele:
            # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
            return None

    assert presort is not None
    if presort:
        # '''
        # Possible advantages: 1> saving time for evaluation;
        # 2> saving time for some models, say the ones need optimal ranking
        # '''
        des_inds = np_arg_shuffle_ties(std_label_vec, descending=True)  # sampling by shuffling ties
        feature_mat, std_label_vec = feature_mat[des_inds], std_label_vec[des_inds]
        # '''
        # if list_docids is None:
        #     list_docids = None
        # else:
        #     list_docids = []
        #     for ind in des_inds:
        #         list_docids.append(list_docids[ind])
        # '''
    return (qid, feature_mat, std_label_vec)


def get_scaler_setting(data_id, scaler_id=None):
    # """
    # A default scaler-setting for loading a dataset
    # :param data_id:
    # :param grid_search: used for grid-search
    # :return:
    # """
    ''' According to {Introducing {LETOR} 4.0 Datasets}, "QueryLevelNorm version: Conduct query level normalization based on data in MIN version. This data can be directly used for learning. We further provide 5 fold partitions of this version for cross fold validation".
     --> Thus there is no need to perform query_level_scale again for {MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi}
     --> But for {MSLRWEB10K | MSLRWEB30K}, the query-level normalization is ## not conducted yet##.
     --> For {Yahoo_LTR_Set_1 | Yahoo_LTR_Set_1 }, the query-level normalization is already conducted.
     --> For Istella! LETOR, the query-level normalization is not conducted yet.
         We note that ISTELLA contains extremely large features, e.g., 1.79769313486e+308, we replace features of this kind with a constant 1000000.
    '''
    if scaler_id is None:
        if data_id in MSLRWEB or data_id in ISTELLA_LTR:
            scale_data = True
            scaler_id = 'SLog1P'  # ['MinMaxScaler', 'StandardScaler']
            scaler_level = 'QUERY'  # SCALER_LEVEL = ['QUERY', 'DATASET']
        else:
            scale_data = False
            scaler_id = None
            scaler_level = None
    else:
        scale_data = True
        scaler_level = 'QUERY'

    return scale_data, scaler_id, scaler_level


def iter_queries(in_file, presort=None, data_dict=None, scale_data=None, scaler_id=None, perquery_file=None,
                 buffer=True):
    # '''
    # Transforms an iterator of rows to an iterator of queries (i.e., a unit of all the documents and labels associated
    # with the same query). Each query is represented by a (qid, feature_mat, std_label_vec) tuple.
    # :param in_file:
    # :param has_comment:
    # :param query_level_scale: perform query-level scaling, say normalization
    # :param scaler: MinMaxScaler | RobustScaler
    # :param unknown_as_zero: if not labled, regard the relevance degree as zero
    # :return:
    # '''
    assert presort is not None
    if os.path.exists(perquery_file): return pickle_load(perquery_file)

    if scale_data: scaler = get_scaler(scaler_id=scaler_id)
    min_docs, min_rele = data_dict['min_docs'], data_dict['min_rele']
    unknown_as_zero, binary_rele, has_comment = data_dict['unknown_as_zero'], data_dict['binary_rele'], data_dict[
        'has_comment']

    clip_query = False
    if min_rele is not None and min_rele > 0:
        clip_query = True
    if min_docs is not None and min_docs > 0:
        clip_query = True

    list_Qs = []
    print(in_file)
    with open(in_file, encoding='iso-8859-1') as file_obj:
        dict_data = dict()
        if has_comment:
            all_features_mat, all_labels_vec, qids, docids = parse_letor(file_obj.readlines(), has_comment=True)

            for i in range(len(qids)):
                f_vec = all_features_mat[i, :]
                std_s = all_labels_vec[i]
                qid = qids[i]
                docid = docids[i]

                if qid in dict_data:
                    dict_data[qid].append((std_s, docid, f_vec))
                else:
                    dict_data[qid] = [(std_s, docid, f_vec)]

            del all_features_mat
            # unique qids
            seen = set()
            seen_add = seen.add
            # sequential unique id
            qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

            for qid in qids_unique:
                tmp = list(zip(*dict_data[qid]))

                list_labels_per_q = tmp[0]
                if data_dict['data_id'] in MSLETOR_LIST:
                    ''' convert the original rank-position into grade-labels '''
                    ranking_size = len(list_labels_per_q)
                    list_labels_per_q = [ranking_size - r for r in list_labels_per_q]

                # list_docids_per_q = tmp[1]
                list_features_per_q = tmp[2]
                feature_mat = np.vstack(list_features_per_q)

                if scale_data:
                    if data_dict['data_id'] in ISTELLA_LTR:
                        # due to the possible extremely large features, e.g., 1.79769313486e+308
                        feature_mat = scaler.fit_transform(np.clip(feature_mat, a_min=None, a_max=ISTELLA_MAX))
                    else:
                        feature_mat = scaler.fit_transform(feature_mat)

                Q = clip_query_data(qid=qid, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q),
                                    binary_rele=binary_rele, unknown_as_zero=unknown_as_zero, clip_query=clip_query,
                                    min_docs=min_docs, min_rele=min_rele, presort=presort)
                if Q is not None:
                    list_Qs.append(Q)
        else:
            if data_dict['data_id'] in YAHOO_LTR:
                all_features_mat, all_labels_vec, qids = parse_letor(file_obj.readlines(), has_comment=False,
                                                                     one_indexed=False)
            else:
                all_features_mat, all_labels_vec, qids = parse_letor(file_obj.readlines(), has_comment=False)

            for i in range(len(qids)):
                f_vec = all_features_mat[i, :]
                std_s = all_labels_vec[i]
                qid = qids[i]

                if qid in dict_data:
                    dict_data[qid].append((std_s, f_vec))
                else:
                    dict_data[qid] = [(std_s, f_vec)]

            del all_features_mat
            # unique qids
            seen = set()
            seen_add = seen.add
            # sequential unique id
            qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

            for qid in qids_unique:
                tmp = list(zip(*dict_data[qid]))
                list_labels_per_q = tmp[0]
                if data_dict['data_id'] in MSLETOR_LIST:
                    ''' convert the original rank-position into grade-labels '''
                    ranking_size = len(list_labels_per_q)
                    list_labels_per_q = [ranking_size - r for r in list_labels_per_q]

                list_features_per_q = tmp[1]
                feature_mat = np.vstack(list_features_per_q)

                if scale_data:
                    if data_dict['data_id'] in ISTELLA_LTR:
                        # due to the possible extremely large features, e.g., 1.79769313486e+308
                        feature_mat = scaler.fit_transform(np.clip(feature_mat, a_min=None, a_max=ISTELLA_MAX))
                    else:
                        feature_mat = scaler.fit_transform(feature_mat)

                Q = clip_query_data(qid=qid, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q),
                                    binary_rele=binary_rele, unknown_as_zero=unknown_as_zero, clip_query=clip_query,
                                    min_docs=min_docs, min_rele=min_rele, presort=presort)
                if Q is not None:
                    list_Qs.append(Q)

    if buffer:
        assert perquery_file is not None
        parent_dir = Path(perquery_file).parent
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        pickle_save(list_Qs, file=perquery_file)

    return list_Qs


## ---------------------------------------------------- ##

class LTRDataset(data.Dataset):
    # """
    # Loading the specified dataset as torch.utils.data.Dataset.
    # We assume that checking the meaningfulness of given loading-setting is conducted beforehand.
    # """
    def __init__(self, split_type, file, data_id=None, data_dict=None, eval_dict=None, presort=True, hot=False,
                 buffer=False, batch_size=1):
        assert data_id is not None or data_dict is not None
        if data_dict is None: data_dict = self.get_default_data_dict(data_id=data_id, batch_size=batch_size)

        self.hot = hot

        ''' data property '''
        self.label_type = data_dict['label_type']

        ''' split-specific settings '''
        self.split_type = split_type
        self.presort = presort
        self.data_id = data_dict['data_id']

        if data_dict['data_id'] in MSLETOR or data_dict['data_id'] in MSLRWEB \
                or data_dict['data_id'] in YAHOO_LTR or data_dict['data_id'] in YAHOO_LTR_5Fold \
                or data_dict['data_id'] in ISTELLA_LTR \
                or data_dict['data_id'] == 'IRGAN_MQ2008_Semi':  # supported datasets

            perquery_file = get_buffer_file_name(data_id=data_id, file=file, data_dict=data_dict, presort=self.presort)

            if hot:
                torch_perquery_file = perquery_file.replace('.np', '_Hot.torch')
            else:
                torch_perquery_file = perquery_file.replace('.np', '.torch')

            if eval_dict is not None:
                mask_label, mask_ratio, mask_type = eval_dict['mask_label'], eval_dict['mask_ratio'], eval_dict[
                    'mask_type']
                if mask_label:
                    mask_label_str = '_'.join([mask_type, 'Ratio', '{:,g}'.format(mask_ratio)])
                    torch_perquery_file = torch_perquery_file.replace('.torch', '_' + mask_label_str + '.torch')
            else:
                mask_label = False

            if os.path.exists(torch_perquery_file):
                st.write('loading buffered file ...')
                self.list_torch_Qs = pickle_load(torch_perquery_file)
            else:
                self.list_torch_Qs = []

                scale_data = data_dict['scale_data']
                scaler_id = data_dict['scaler_id'] if 'scaler_id' in data_dict else None
                list_Qs = iter_queries(in_file=file, presort=self.presort, data_dict=data_dict, scale_data=scale_data,
                                       scaler_id=scaler_id, perquery_file=perquery_file, buffer=buffer)

                list_inds = list(range(len(list_Qs)))
                for ind in list_inds:
                    qid, doc_reprs, doc_labels = list_Qs[ind]

                    torch_q_doc_vectors = torch.from_numpy(doc_reprs).type(torch.FloatTensor)
                    # torch_q_doc_vectors = torch.unsqueeze(torch_q_doc_vectors, dim=0)  # a default batch size of 1

                    torch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)
                    # torch_std_labels = torch.unsqueeze(torch_std_labels, dim=0) # a default batch size of 1

                    self.list_torch_Qs.append((qid, torch_q_doc_vectors, torch_std_labels))
                # buffer
                # print('Num of q:', len(self.list_torch_Qs))
                if buffer:
                    parent_dir = Path(torch_perquery_file).parent
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir)
                    pickle_save(self.list_torch_Qs, torch_perquery_file)
        else:
            raise NotImplementedError

    def get_default_data_dict(self, data_id, scaler_id=None, batch_size=None):
        # ''' a default setting for loading a dataset '''
        min_docs = 1
        min_rele = 1  # with -1, it means that we don't care with dumb queries that has no relevant documents. Say, for checking the statistics of an original dataset
        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=data_id, scaler_id=scaler_id)

        train_presort = False if data_id in MSLETOR_SEMI else True

        batch_size = 10 if batch_size is None else batch_size

        data_dict = dict(data_id=data_id, min_docs=min_docs, min_rele=min_rele, binary_rele=False,
                         unknown_as_zero=False,
                         train_presort=train_presort, validation_presort=True, test_presort=True,
                         train_batch_size=batch_size, validation_batch_size=batch_size, test_batch_size=batch_size,
                         scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

        data_meta = get_data_meta(data_id=data_id)
        data_dict.update(data_meta)

        return data_dict

    def __len__(self):
        return len(self.list_torch_Qs)

    def __getitem__(self, index):
        qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
        return qid, torch_batch_rankings, torch_batch_std_labels

    def iter_hot(self):
        list_inds = list(range(len(self.list_torch_Qs)))
        for ind in list_inds:
            qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts = \
            self.list_torch_Qs[ind]
            yield qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts


## Customize Sampler for Batch Processing ##

def pre_allocate_batch(dict_univ_bin, num_docs_per_batch):
    # '''
    # Based on the expected number of documents to process within a single batch, we merge the queries that have the same number of documents to form a batch
    # @param dict_univ_bin: [unique_value, bin of index]
    # @param num_docs_per_batch:
    # @return:
    # '''
    list_batch_inds = []

    if 1 == num_docs_per_batch:  # a simple but time-consuming per-query processing, namely the batch_size is always one
        for univ in dict_univ_bin:
            bin = dict_univ_bin[univ]
            for index in bin:
                single_ind_as_batch = [index]
                list_batch_inds.append(single_ind_as_batch)

        return list_batch_inds
    else:
        for univ in dict_univ_bin:
            bin = dict_univ_bin[univ]
            bin_length = len(bin)

            if univ * bin_length < num_docs_per_batch:  # merge all queries as one batch
                list_batch_inds.append(bin)
            else:
                if univ < num_docs_per_batch:  # split with an approximate value
                    num_inds_per_batch = num_docs_per_batch // univ
                    for i in range(0, bin_length, num_inds_per_batch):
                        sub_bin = bin[i: min(i + num_inds_per_batch, bin_length)]
                        list_batch_inds.append(sub_bin)
                else:  # one single query as a batch
                    for index in bin:
                        single_ind_as_batch = [index]
                        list_batch_inds.append(single_ind_as_batch)

        return list_batch_inds


class LETORSampler(data.Sampler):
    # '''
    # Customized sampler for LETOR datasets based on the observation that:
    # though the number of documents per query may differ, there are many queries that have the same number of documents, especially with a big dataset.
    # '''
    def __init__(self, data_source, rough_batch_size=None):
        list_num_docs = []
        for qid, torch_batch_rankings, torch_batch_std_labels in data_source:
            list_num_docs.append(torch_batch_std_labels.size(0))

        dict_univ_bin = {}
        for ind, univ in enumerate(list_num_docs):
            if univ in dict_univ_bin:
                dict_univ_bin[univ].append(ind)
            else:
                bin = [ind]
                dict_univ_bin[univ] = bin

        self.list_batch_inds = pre_allocate_batch(dict_univ_bin=dict_univ_bin, num_docs_per_batch=rough_batch_size)

    def __iter__(self):
        for batch_inds in self.list_batch_inds:
            yield batch_inds


def load_data(data_id, file, split_type, batch_size):
    _ltr_data = LTRDataset(data_id=data_id, file=file, split_type=split_type, batch_size=batch_size)
    letor_sampler = LETORSampler(data_source=_ltr_data, rough_batch_size=batch_size)
    ltr_data = torch.utils.data.DataLoader(_ltr_data, batch_sampler=letor_sampler, num_workers=0)
    return ltr_data


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
