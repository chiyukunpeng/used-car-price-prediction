import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def tree_feature(train_data, test_data):
    # 处理目标值长尾分布问题
    train_data['price'] = np.log1p(train_data['price'])

    # 合并数据以便后续操作
    data = pd.concat([train_data, test_data], ignore_index=True)

    # 数据预处理
    data = data_preprocessing(data)

    # 特征工程
    data = feature_engineering(data, train_data)

    # 筛选特征
    data = select_feature(data)

    return data


def data_preprocessing(data):
    # 处理无用值
    data['name_count'] = data.groupby(['name'])['SaleID'].transform('count')
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
    data['notRepairedDamage'] = data['notRepairedDamage'].astype('str').apply(lambda x: x if x != '-' else None).astype(
        'float32')

    return data


def feature_engineering(data, train_data):
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

    # 地区类特征
    data['regionCode_count'] = data.groupby(['regionCode'])['SaleID'].transform('count')
    data['city'] = data['regionCode'].apply(lambda x: str(x)[:2])

    # 可分类特征，进行分桶
    bin = [i * 10 for i in range(31)]
    data['power_bin'] = pd.cut(data['power'], bin, labels=False)
    tong = data[['power_bin', 'power']].head()

    bin = [i * 10 for i in range(24)]
    data['model_bin'] = pd.cut(data['model'], bin, labels=False)
    tong = data[['model_bin', 'model']].head()

    # 可分类特征组合,与目标特征price组合
    data = feature_merge(data, train_data, 'regionCode')
    data = feature_merge(data, train_data, 'brand')
    data = feature_merge(data, train_data, 'model')
    data = feature_merge(data, train_data, 'kilometer')
    data = feature_merge(data, train_data, 'bodyType')
    data = feature_merge(data, train_data, 'fuelType')

    # 其他可分类特征组合
    feat1 = 'regionCode'
    train_gb = data.groupby(feat1)
    infos_dic = {}
    for key, value in train_gb:
        info_dic = {}
        value = value[value['car_age_day'] > 0]

        info_dic[feat1 + '_days_max'] = value.car_age_day.max()
        info_dic[feat1 + '_days_min'] = value.car_age_day.min()
        info_dic[feat1 + '_days_mean'] = value.car_age_day.mean()
        info_dic[feat1 + '_days_std'] = value.car_age_day.std()
        info_dic[feat1 + '_days_sum'] = value.car_age_day.sum()
        info_dic[feat1 + '_days_median'] = value.car_age_day.median()

        infos_dic[key] = info_dic
    df = pd.DataFrame(infos_dic).T.reset_index().rename(columns={"index": feat1})
    data = data.merge(df, how='left', on=feat1)

    train_gb = data.groupby(feat1)
    infos_dic = {}
    for key, value in train_gb:
        info_dic = {}
        value = value[value['power'] > 0]

        info_dic[feat1 + '_days_max'] = value.power.max()
        info_dic[feat1 + '_days_min'] = value.power.min()
        info_dic[feat1 + '_days_mean'] = value.power.mean()
        info_dic[feat1 + '_days_std'] = value.power.std()
        info_dic[feat1 + '_days_sum'] = value.power.sum()
        info_dic[feat1 + '_days_median'] = value.power.median()

        infos_dic[key] = info_dic
    df = pd.DataFrame(infos_dic).T.reset_index().rename(columns={"index": feat1})
    data = data.merge(df, how='left', on=feat1)

    # 匿名特征组合
    feat2 = 'v_3'
    train_gb = data.groupby(feat1)
    infos_dic = {}
    for key, value in train_gb:
        info_dic = {}
        value = value[value[feat2] > -10000000]

        info_dic[feat1 + '_' + feat2 + '_max'] = value.v_3.max()
        info_dic[feat1 + '_' + feat2 + '_min'] = value.v_3.min()
        info_dic[feat1 + '_' + feat2 + '_mean'] = value.v_3.mean()
        info_dic[feat1 + '_' + feat2 + '_std'] = value.v_3.std()
        info_dic[feat1 + '_' + feat2 + '_sum'] = value.v_3.sum()
        info_dic[feat1 + '_' + feat2 + '_median'] = value.v_3.median()

        infos_dic[key] = info_dic
    df = pd.DataFrame(infos_dic).T.reset_index().rename(columns={'index': feat1})
    data = data.merge(df, how='left', on=feat1)

    feat3 = 'v_0'
    train_gb = data.groupby(feat1)
    infos_dic = {}
    for key, value in train_gb:
        info_dic = {}
        value = value[value[feat3] > -10000000]

        info_dic[feat1 + '_' + feat3 + '_max'] = value.v_0.max()
        info_dic[feat1 + '_' + feat3 + '_min'] = value.v_0.min()
        info_dic[feat1 + '_' + feat3 + '_mean'] = value.v_0.mean()
        info_dic[feat1 + '_' + feat3 + '_std'] = value.v_0.std()
        info_dic[feat1 + '_' + feat3 + '_sum'] = value.v_0.sum()
        info_dic[feat1 + '_' + feat3 + '_median'] = value.v_0.median()

        infos_dic[key] = info_dic
    df = pd.DataFrame(infos_dic).T.reset_index().rename(columns={'index': feat1})
    data = data.merge(df, how='left', on=feat1)

    # 特征交叉，针对匿名特征及重要性高的可分类特征
    for i in range(15):
        for j in range(15):
            data['new' + str(i) + '*' + str(j)] = data['v_' + str(i)] * data['v_' + str(j)]

    for i in range(15):
        for j in range(15):
            data['new' + str(i) + '+' + str(j)] = data['v_' + str(i)] + data['v_' + str(j)]

    for i in range(15):
        data['new' + str(i) + '*power'] = data['v_' + str(i)] * data['power']

    for i in range(15):
        data['new' + str(i) + '*day'] = data['v_' + str(i)] * data['car_age_day']

    for i in range(15):
        data['new' + str(i) + '*year'] = data['v_' + str(i)] * data['car_age_year']

    return data


def date_process(date):
    year = int(str(date)[:4])
    month = int(str(date)[4:6])
    day = int(str(date)[6:8])
    if month < 1:
        month = 1
    date = datetime(year, month, day)

    return date


def feature_merge(data, train_data, feature):
    train_gb = train_data.groupby(str(feature))
    infos_dic = {}

    for key, value in train_gb:
        info_dic = {}
        value = value[value['price'] > 0]

        info_dic[str(feature) + '_amount'] = len(value)
        info_dic[str(feature) + '_price_max'] = value.price.max()
        info_dic[str(feature) + '_price_min'] = value.price.min()
        info_dic[str(feature) + '_price_median'] = value.price.median()
        info_dic[str(feature) + '_price_sum'] = value.price.sum()
        info_dic[str(feature) + '_price_std'] = value.price.std()
        info_dic[str(feature) + '_price_mean'] = value.price.mean()
        info_dic[str(feature) + '_price_skew'] = value.price.skew()
        info_dic[str(feature) + '_price_kurt'] = value.price.kurt()
        info_dic[str(feature) + '_mad'] = value.price.mad()

        infos_dic[key] = info_dic

    df = pd.DataFrame(infos_dic).T.reset_index().rename(columns={"index": str(feature)})
    data = data.merge(df, how='left', on=str(feature))

    return data


def select_feature(data):
    # 数值型特征
    numerical_cols = data.select_dtypes(exclude='object').columns
    tree_feature_list = ['model_power_sum', 'price', 'SaleID',
                         'model_power_std', 'model_power_median', 'model_power_max',
                         'brand_price_max', 'brand_price_median',
                         'brand_price_sum', 'brand_price_std',
                         'model_days_sum',
                         'model_days_std', 'model_days_median', 'model_days_max', 'model_bin', 'model_amount',
                         'model_price_max', 'model_price_median',
                         'model_price_min', 'model_price_sum', 'model_price_std',
                         'model_price_mean', 'bodyType', 'model', 'brand', 'fuelType', 'gearbox', 'power', 'kilometer',
                         'notRepairedDamage', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9',
                         'v_10',
                         'v_11', 'v_12', 'v_13', 'v_14', 'name_count', 'regDate_year', 'car_age_day', 'car_age_year',
                         'power_bin', 'fuelType', 'gearbox', 'kilometer', 'notRepairedDamage', 'v_0', 'v_3', 'v_6',
                         'v_10',
                         'name_count', 'car_age_day', 'new3*3', 'new12*14', 'new2*14', 'new14*14']
    anonymous_feature_list = ['new14+6', 'new13+6', 'new0+12', 'new9+11', 'v_3', 'new11+10', 'new10+14', 'new12+4',
                              'new3+4', 'new11+11', 'new13+3', 'new8+1', 'new1+7', 'new11+14', 'new8+13', 'v_8', 'v_0',
                              'new3+5', 'new2+9', 'new9+2', 'new0+11', 'new13+7', 'new8+11', 'new5+12', 'new10+10',
                              'new13+8', 'new11+13', 'new7+9', 'v_1', 'new7+4', 'new13+4', 'v_7', 'new5+6', 'new7+3',
                              'new9+10', 'new11+12', 'new0+5', 'new4+13', 'new8+0', 'new0+7', 'new12+8', 'new10+8',
                              'new13+14', 'new5+7', 'new2+7', 'v_4', 'v_10', 'new4+8', 'new8+14', 'new5+9', 'new9+13',
                              'new2+12', 'new5+8', 'new3+12', 'new0+10', 'new9+0', 'new1+11', 'new8+4', 'new11+8',
                              'new1+1',
                              'new10+5', 'new8+2', 'new6+1', 'new2+1', 'new1+12', 'new2+5', 'new0+14', 'new4+7',
                              'new14+9',
                              'new0+2', 'new4+1', 'new7+11', 'new13+10', 'new6+3', 'new1+10', 'v_9', 'new3+6',
                              'new12+1',
                              'new9+3', 'new4+5', 'new12+9', 'new3+8', 'new0+8', 'new1+8', 'new1+6', 'new10+9',
                              'new5+4',
                              'new13+1', 'new3+7', 'new6+4', 'new6+7', 'new13+0', 'new1+14', 'new3+11', 'new6+8',
                              'new0+9',
                              'new2+14', 'new6+2', 'new12+12', 'new7+12', 'new12+6', 'new12+14', 'new4+10', 'new2+4',
                              'new6+0', 'new3+9', 'new2+8', 'new6+11', 'new3+10', 'new7+0', 'v_11', 'new1+3', 'new8+3',
                              'new12+13', 'new1+9', 'new10+13', 'new5+10', 'new2+2', 'new6+9', 'new7+10', 'new0+0',
                              'new11+7', 'new2+13', 'new11+1', 'new5+11', 'new4+6', 'new12+2', 'new4+4', 'new6+14',
                              'new0+1', 'new4+14', 'v_5', 'new4+11', 'v_6', 'new0+4', 'new1+5', 'new3+14', 'new2+10',
                              'new9+4', 'new2+6', 'new14+14', 'new11+6', 'new9+1', 'new3+13', 'new13+13', 'new10+6',
                              'new2+3', 'new2+11', 'new1+4', 'v_2', 'new5+13', 'new4+2', 'new0+6', 'new7+13', 'new8+9',
                              'new9+12', 'new0+13', 'new10+12', 'new5+14', 'new6+10', 'new10+7', 'v_13', 'new5+2',
                              'new6+13', 'new9+14', 'new13+9', 'new14+7', 'new8+12', 'new3+3', 'new6+12', 'v_12',
                              'new14+4',
                              'new11+9', 'new12+7', 'new4+9', 'new4+12', 'new1+13', 'new0+3', 'new8+10', 'new13+11',
                              'new7+8', 'new7+14', 'v_14', 'new10+11', 'new14+8', 'new1+2']
    for i in range(15):
        for j in range(15):
            tree_feature_list.append('new' + str(i) + '+' + str(j))

    feature_cols = [col for col in numerical_cols if col in tree_feature_list]
    feature_cols = [col for col in feature_cols if col not in anonymous_feature_list]
    data = data[feature_cols]

    return data


