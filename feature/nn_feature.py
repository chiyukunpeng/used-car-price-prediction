import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def nn_feature(train_data, test_data):
    # 处理目标值长尾分布问题
    train_data['price'] = np.log1p(train_data['price'])
    # 合并数据以便后续操作
    data = pd.concat([train_data, test_data], ignore_index=True)
    # 选择特征
    feature = ['model', 'brand', 'bodyType', 'fuelType', 'kilometer', 'notRepairedDamage', 'power', 'regDate_month',
               'creatDate_year', 'creatDate_month', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
               'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14', 'car_age_day', 'car_age_year',
               'regDate_year', 'name_count']
    # 数据预处理
    data = data_preprocessing(data)

    # 特征工程
    data = feature_engineering(data, feature)

    return data


def data_preprocessing(data):
    # 对name进行挖掘
    data['name_count'] = data.groupby(['name'])['SaleID'].transform('count')
    # 处理无用值
    del data['name']
    data.drop(data[data['seller'] == 1].index, inplace=True)
    del data['offerType']
    del data['seller']

    # 处理缺失值,填充众数
    data['fuelType'] = data['fuelType'].fillna(0)
    data['gearbox'] = data['gearbox'].fillna(0)
    data['bodyType'] = data['bodyType'].fillna(0)
    data['model'] = data['model'].fillna(0)

    # 处理异常值
    data['power'] = data['power'].map(lambda x: 600 if x > 600 else x)
    data.replace(to_replace='-', value=0.5, inplace=True)  # NN标签转换，不确定值位于确定值中间
    data['notRepairedDamage'] = LabelEncoder().fit_transform(data['notRepairedDamage'].astype(str))

    return data


def feature_engineering(data, feature):
    # 时间类特征
    data['regDate'] = data['regDate'].apply(date_process)
    data['creatDate'] = data['creatDate'].apply(date_process)
    data['regDate_year'] = data['regDate'].dt.year
    data['regDate_month'] = data['regDate'].dt.month
    data['regDate_day'] = data['regDate'].dt.day
    data['creatDate_year'] = data['creatDate'].dt.year
    data['creatDate_month'] = data['creatDate'].dt.month
    data['creatDate_day'] = data['creatDate'].dt.day
    data['car_age_day'] = (data['creatDate'] - data['regDate']).dt.days
    data['car_age_year'] = round(data['car_age_day'] / 365, 1)

    # 特征归一化
    scaler = MinMaxScaler()
    scaler.fit(data[feature].values)

    return data


def date_process(date):
    year = int(str(date)[:4])
    month = int(str(date)[4:6])
    day = int(str(date)[6:8])
    if month < 1:
        month = 1
    date = datetime(year, month, day)

    return date


