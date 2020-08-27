##  Introduction

This project is a contest in tianchi of Alibaba. It aims to predict price of used car based on 200k datasets. Our method uses mixed models based on EDA and feature engineering. Expriments show that our method outperforms most approaches. 

For more information, please see [CSDN](https://blog.csdn.net/chiyukunpeng/article/details/108253390) 

## Quick Start

#### 1 Requirements
Python 3.7 or later with all requirements.txt dependencies installed, including tensorflow>=2.1.0. To install run:
```
$ pip install -r requirements.txt
```

#### 2 Organize directory
```
---code
    |--- main.py
---data
    |--- used_car_train_20200313.csv
    |--- used_car_testB_20200421.csv
---feature
    |--- tree_feature.py
    |--- nn_feature.py
---model
    |--- model.py
---prediction_result
---user_data
```

#### 3 Train and predict
Please run:
```
$ python main.py
```

## Citation
If you find this project useful for your research, please cite:
```
@{used car price prediction project,
author = {chen peng},
title = {UCPP},
website = {https://github.com/chiyukunpeng/used-car-price-prediction},
month = {August},
year = {2020}
}
```
