"""
A simple implementation of Bayesian Classifier
@data: 2019.12.11
@author: Tingyu Mo
"""
import pandas as pd
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score

from BayesClassfier import BayesianClassifier


def data_generator(data_path,risk_path):
    '''
    load data from .dat files and make preprocessing 
    '''
    risk_table = pd.read_csv(risk_path,header=None,encoding='utf-8',delimiter="\n",quoting=csv.QUOTE_NONE)
    risk_table = risk_table.values
    risk_table = risk_table.tolist()
    risk_list = list()
    for dataterm in risk_table:
        data_list = dataterm[0].split(',')
        data_list = [float(i)for i in data_list]
        risk_list.append([i for i in data_list])#risk

    df = pd.read_csv(data_path,header=None,encoding='utf-8',delimiter="\n",quoting=csv.QUOTE_NONE)
    df = df.iloc[3:,:] #去除头部说明信息
    data = df.values#将dataframe转成ndarray 内容为str
    data = data.tolist()#转成list
    features = []
    #数据集内容划分 分为features部分和labels部分
    print("preprocessing datasets!")
    for dataterm in tqdm(data):
        data_list = dataterm[0].split(',')
        data_list = [float(i)for i in data_list]
        features.append([i for i in data_list])#除label以外所有的features
    return np.array(features) ,np.array(risk_list)

def experiment_build(data_path,risk_path):
    Cell_data,risk_table = data_generator(data_path,risk_path)
    Cell_data = np.linspace(-3.09,3.19,250)
    prior_distribution = {"1":0.9,"-1":0.1}
    params_dict = {"1":(-2,0.25),"-1":(2,4)}
    BF = BayesianClassifier("MRD","max-likelihood")
    BF.prior_distribution = prior_distribution
    BF.params_dict = params_dict
    BF.class_name = list(params_dict.keys())

    result = BF.multi_predict(x_tensor = Cell_data,risk = risk_table)
    print(result)
    BF.viz_risk(Cell_data)
    return 0

if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion/Exp1-2'
    os.chdir(Root_dir)
    datasets_path = os.path.join(Root_dir,'datasets')
    data_path = os.path.join(datasets_path,"cell.dat")
    risk_path = os.path.join(datasets_path,"risk.csv")
    experiment_build(data_path,risk_path)