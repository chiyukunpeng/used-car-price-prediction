import pandas as pd
import numpy as np
import argparse
from feature.tree_feature import tree_feature
from feature.nn_feature import nn_feature
from model.model import final_model
import warnings

warnings.filterwarnings('ignore')


def main():
    train_data = pd.read_csv(args.train_data, sep=' ')
    test_data = pd.read_csv(args.test_data, sep=' ')

    # 创建树模型数据
    print('tree model data making...')
    tree_data = tree_feature(train_data, test_data)
    print("tree model data shape:", tree_data.shape)
    tree_train_num = int(tree_data.shape[0] - 50000)
    tree_data[:tree_train_num].to_csv(args.train_tree, index=0, sep=' ')
    tree_data[tree_train_num:tree_train_num + 50000].to_csv(args.test_tree, index=0, sep=' ')
    print("tree model data made")

    # 创建nn模型数据
    print("nn model data making...")
    nn_data = nn_feature(train_data, test_data)
    print("nn model data shape:", nn_data.shape)
    nn_train_num = int(nn_data.shape[0] - 50000)
    nn_data[:nn_train_num].to_csv(args.train_nn, index=0, sep=' ')
    nn_data[nn_train_num:nn_train_num + 50000].to_csv(args.test_nn, index=0, sep=' ')
    print("nn model data made")

    # 构建树模型训练与测试样本
    tree_train_data = pd.read_csv(args.train_tree, sep=' ')
    tree_test_data = pd.read_csv(args.test_tree, sep=' ')
    tree_numerical_cols = tree_train_data.columns
    tree_feature_cols = [col for col in tree_numerical_cols if col not in ['price', 'SaleID']]
    x_train_tree = np.array(tree_train_data[tree_feature_cols])
    x_test_tree = np.array(tree_test_data[tree_feature_cols])
    y_train_tree = np.array(tree_train_data['price'])
    print('x_train_tree shape:{}  x_test_tree shape:{}'.format(x_train_tree.shape, x_test_tree.shape))

    # 构建nn模型训练与测试样本
    nn_train_data = pd.read_csv(args.train_nn, sep=' ')
    nn_test_data = pd.read_csv(args.test_nn, sep=' ')
    nn_numerical_cols = nn_train_data.columns
    nn_feature_cols = [col for col in nn_numerical_cols if col not in ['price', 'SaleID']]
    x_train_nn = np.array(nn_train_data[nn_feature_cols])
    x_test_nn = np.array(nn_test_data[nn_feature_cols])
    y_train_nn = np.array(nn_train_data['price'])
    print('x_train_nn shape:{}  x_test_tree shape:{}'.format(x_train_nn.shape, x_test_nn.shape))

    # 模型预测
    predictions = final_model(x_train_tree, x_test_tree, y_train_tree, x_train_nn, x_test_nn, y_train_nn)
    df = pd.DataFrame()
    df['SaleID'] = nn_test_data.SaleID
    predictions[predictions < 0] = 0
    df['price'] = predictions
    df.to_csv(args.precitions, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',
                        type=str,
                        default='../data/used_car_train_20200313.csv',
                        help='path to original train data'
                        )
    parser.add_argument('--test_data',
                        type=str,
                        default='../data/used_car_testB_20200421.csv',
                        help='path to original test data'
                        )
    parser.add_argument('--train_tree',
                        type=str,
                        default='../user_data/train_tree.csv',
                        help='path to tree train data'
                        )
    parser.add_argument('--test_tree',
                        type=str,
                        default='../user_data/test_tree.csv',
                        help='path to tree test data'
                        )
    parser.add_argument('--train_nn',
                        type=str,
                        default='../user_data/train_nn.csv',
                        help='path to nn train data'
                        )
    parser.add_argument('--test_nn',
                        type=str,
                        default='../user_data/test_nn.csv',
                        help='path to nn test data'
                        )
    parser.add_argument('--predictions',
                        type=str,
                        default='../prediction_result/predictions.csv',
                        help='path to predictions result'
                        )
    args = parser.parse_args()
    main()
