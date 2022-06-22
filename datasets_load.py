import streamlit as st
import os
import torch
from pathlib import Path
from torch.utils import data


from pre_datasets import \
    pickle_load, get_buffer_file_name, MSLETOR, YAHOO_LTR, ISTELLA_LTR, MSLRWEB, pickle_save, iter_queries, \
    YAHOO_LTR_5Fold, get_scaler_setting, MSLETOR_SEMI, get_data_meta

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
