import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from preprocessing.inputs import SparseFeat, DenseFeat, get_feature_names
from model.wdm import WideDeep


if __name__ == '__main__':
    # 1.load data
    data = pd.read_csv('./data/criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 2.preprocessing
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 3.generate feature columns
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]
    dense_feature_columns = [DenseFeat(feat, dimension=1) for feat in dense_features]

    linear_feature_columns = sparse_feature_columns + dense_feature_columns
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns

    feature_names = get_feature_names(linear_feature_columns)

    # 4.Generate the training samples and train the model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = WideDeep(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'], )

    model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
