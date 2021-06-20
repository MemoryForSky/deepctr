import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict, namedtuple
from layers.sequence import SequencePoolingLayer
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype',
                                           'embedding_name', 'group_name'])):
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype='int32', embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print("Notice! Feature Hashing on the fly currently!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    def __new__(cls, sparsefeat, maxlen, combiner='mean', length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict({feat.embedding_name: nn.Embedding(feat.vocabulary_size,
                                                                      feat.embedding_dim if not linear else 1)
                                    for feat in sparse_feature_columns + varlen_sparse_feature_columns})

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []

    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.embedding_name](
                        features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long())
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0

            emb = SequencePoolingLayer(mode=feat.combiner, support_masking=True, device=device)([seq_emb, seq_mask])

        else:
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()

            emb = SequencePoolingLayer(mode=feat.combiner, support_masking=False, device=device)([seq_emb, seq_length])

        varlen_sparse_embedding_list.append(emb)

    return varlen_sparse_embedding_list


def build_input_features(feature_columns):
    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start+1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def input_from_feature_columns(X, feature_index, feature_columns, embedding_dict, support_dense=True, device='cpu'):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for feat in sparse_feature_columns]

    varlen_sparse_embedding_list = get_varlen_pooling_list(embedding_dict, X, feature_index,
                                                           varlen_sparse_feature_columns, device)

    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list


def compute_input_dim(feature_columns, include_sparse=True, include_dense=True, feature_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
        feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
    if feature_group:
        sparse_input_dim = len(sparse_feature_columns)
    else:
        sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
    input_dim = 0
    if include_sparse:
        input_dim += sparse_input_dim
    if include_dense:
        input_dim += dense_input_dim
    return input_dim


def embedding_lookup(X, sparse_embedding_dict, sparse_input_dict, sparse_feature_columns,
                     return_feat_list=(), mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            lookup_idx = np.array(sparse_input_dict[feature_name])
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
            # TODO: add hash function
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            X[:, lookup_idx[0]:lookup_idx[1]].long())  # (lookup_idx)

    return varlen_embedding_vec_dict


def maxlen_lookup(X, sparse_input_dict, maxlen_column):
    if maxlen_column is None or len(maxlen_column)==0:
        raise ValueError('please add max length column for VarLenSparseFeat of DIN/DIEN input')
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0]:lookup_idx[1]].long()

# if __name__ == '__main__':
#     user_id = SparseFeat('user_id', 1000, embedding_dim=4)
#     score_avg = DenseFeat('score_avg', dimension=1)
#     user_hist = VarLenSparseFeat(SparseFeat('user_hist', 1000, embedding_dim=4), 666)

