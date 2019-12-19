"""
A simple implementation of Bayesian Classifier
@data: 2019.12.15
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
# import torch

class BayesianClassifier():
    '''
    A simple implementation of Bayesian Classifier
    '''
    def __init__(self,decision_method = 'MSD',CCP_estimate_method="max-likelihood"):
        self.class_name = [] #类名
        self.decision_method = decision_method
        self.CCP_estimate_method = CCP_estimate_method
        self.prior_distribution = {}
        self.params_dict = {} #存放各类概率密度函数的参数 {'classname',(mu,std)}
        self.CCPPF = {}  #类条件概率密度函数 {'classname',p(x|w)}
        self.posterior_distribution = {} 
        self.risk_list = []#存放数据流计算的条件风险列表
        self.kN_CCPPF = {}
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def data_generator(self,data_path):
        '''
        load data from .dat files and make preprocessing 
        '''
        df = pd.read_csv(data_path,header=None,encoding='utf-8',delimiter="\t",quoting=csv.QUOTE_NONE)
        df = df.iloc[3:,:] #去除头部说明信息
        data = df.values#将dataframe转成ndarray 内容为str
        data = data.tolist()#转成list
        features = []
        labels = []
        #数据集内容划分 分为features部分和labels部分
        print("preprocessing datasets!")
        for dataterm in tqdm(data):
            data_list = dataterm[0].split(',')
            data_list = [float(i)for i in data_list]
            features.append([i for i in data_list[:-1]])#除label以外所有的features
            labels.append(data_list[-1])
        # self.features,self.labels = np.array(features),np.array(labels)
        return np.array(features),np.array(labels)

    def dataset_split(self,train_data,train_target,test_size = 0.3,random_state=None):
        # x_train,x_test, y_train, y_test = train_test_split(train_data,train_target,test_size=test_size, random_state=random_state)#划分数据集
        return train_test_split(train_data,train_target,test_size=test_size, random_state=random_state)
    
    def risk_table_csv(self,risk_csv):
        '''
        read risk data from csv file return a 2d ndarray,which row contains 
        the risk for each class with repect to the misclassified class.
        '''
        risk_table = []
        with open(risk_csv, "r") as f:    #打开文件
            risk_table = f.read().split('\n')
        class_risk_list = list()
        for class_risk in risk_table:
            class_risk = class_risk.split(",")
            class_risk =  [ float(i) for i in class_risk ]
            class_risk_list.append(class_risk)
        return np.array(class_risk_list)


    def prior_probility(self,features,labels):
        '''
        computed prior_distribution p(x)
        '''
        prior_distribution = dict()
        num_labels = len(labels)
        self.class_name = list()
        print("caculate prior_distribution!")
        for label in tqdm(labels):
            label_str = str(label) #将标签转换为字典的key
            if label_str not in prior_distribution:
                prior_distribution[label_str] = 0.0 #添加新标签
                self.class_name.append(label_str)
            prior_distribution[label_str] += 1.0/num_labels #归一化

        return prior_distribution 
    
    def class_conditional_probability(self,x,x_train,y_train,h1=0.5):
        '''
        computed class_conditional_probability p(x|w)
        '''
        if self.CCP_estimate_method == "max-likelihood":
            if not self.params_dict:
                self.params_dict = self.maximum_likelihood_estimate(x_train,y_train)
            labelname = self.class_name
            CCPPF = {}
            for key in labelname:
                mu,sigma = self.params_dict[key]
                CCPPF[key] = self.Gaussian_distribution(x,mu,sigma)
        if self.CCP_estimate_method == "parzen-window":
            CCPPF = self.parzen_window_estimate(x,x_train,y_train,h1)

        if self.CCP_estimate_method == "k-neighbours":
            if not self.kN_CCPPF:
                N = 500
                for key in self.class_name:
                    self.kN_CCPPF[key] = np.zeros(N)
                self.kN_CCPPF_estimate(x_train,y_train)
            CCPPF = self.kneighbours_estimate(x)

        self.CCPPF = CCPPF
        return CCPPF

    def posterior_probability(self,x,CCPPF,prior):
        '''
        computed posterior_distribution p(w|x)
        '''
        labelname = self.class_name
        posterior_distribution ={}
        for key in labelname:
            #Bayes formula : p(w|x) =^ p(x|w)*p(w)
            posterior_distribution[key] = prior[key]*CCPPF[key]
        self.posterior_distribution = posterior_distribution
        return posterior_distribution
    
    def Gaussian_distribution(self,x,mu,sigma):
        '''
        Caculate the gaussian distribution with specific x value , mean ,standard variance inputs
        '''
        return 1.0/(2*math.pi)*math.exp(-1.0/2*math.pow((x-mu)*1.0/sigma,2))
    
    def caculate_gaussian(self,x,mu,sigma):
        #高斯函数可视化
        # x = np.linspace(-3.09,3.19,50)#从(-1,1)均匀取50个点
        # y = BC.caculate_gaussian(x,1,0.5)
        # plt.plot(x,y)
        # plt.show()
        output = list()
        for point in x:
            output.append(self.Gaussian_distribution(point,mu,sigma))
        return np.array(output)

    def maximum_likelihood_estimate(self,features,labels):
        '''
        According to max likelihood estimate theory,the gaussian distribution 
        probility function's parameters are calculated via following conclusions

        mu equals to the mean of sampls

        std equasl to the std of samples

        return an parameters dictionary containing the parameters of normal distribution E.g {"classname":tuple(mu,std)}

        '''
        labelname = self.class_name
        features_one = features[:,0] #第一个特征
        params_dict = {}
        for key in labelname:
            index = np.argwhere(labels==float(key))#利用标签获取该类的索引
            class_data = features_one[index]#利用该类的索引获取数据
            class_data_mean = np.mean(class_data)#均值
            class_data_std = np.std(class_data)#标准差
            params_dict[key] = (class_data_mean, class_data_std)#{'classname',(mu,std)}

        return params_dict
    
    def parzen_window_estimate(self,x,x_train,y_train,h1=0.5):
        '''
        According to parzen window estimation method,we are able to calculated
        the CCPPF via the following formula:

        p(x|w) = 1/N*sum(exp[-1/2*(x-xi)^2/h1])

        return CCPPF = {"classname":p(x|w)}
        '''
        labelname = self.class_name
        x_train_one = x_train[:,0] #第一个特征
        window_h = self.get_parzenwindow_hN(h1,y_train.size)
        CCPPF = {}
        # print("start calculate CCPPF!\n")
        for key in labelname:
            index = np.argwhere(y_train==float(key))#利用标签获取该类的索引
            class_data = x_train_one[index]#利用该类的索引获取数据
            N = class_data.size
            CCP = 0
            for i in range(N):
                # CCP += self.Gaussian_distribution(x,class_data[i],math.sqrt(window_h))*1.0/(N*window_h) #计算Pn(x)
                CCP += self.Gaussian_distribution(x,class_data[i],window_h)*1.0/(N*window_h) #计算Pn(x)
            CCPPF[key] = CCP
        return CCPPF

    # def parzen_window_function(self,x,class_data,window_h,method = 'GD'):
    #     if method == 'GD':
    #         return self.Gaussian_distribution(x,class_data[i],math.sqrt(window_h))*1.0/(N*window_h)

    def get_parzenwindow_hN(self,h1,N):
        '''
        calculated parzen window's hN ,which is the width of the window
        via VN = hN = h1/sqrt(N)
        '''
        return h1/math.sqrt(1.0*N) 

    def kneighbours_estimate(self,x):
        '''
        According to kneighbours_estimate method,we are able to calculate CCPPF with
        the changable window width , the number of samples in the window is calculated via the following
        formula : kN = sqrt(N)
        '''    
        CCPPF = {}
        x_axis = np.linspace(-3.09,3.19,500)
        for key in self.class_name:
            index_small = np.argwhere(x<x_axis)
            if index_small.size>1:
                index_small = index_small[0]
            index_big = np.argwhere(x>x_axis)
            if index_big.size>1:
                index_big = index_big[-1]
            CCPPF[key] = 1.0*(self.kN_CCPPF[key][index_small]+self.kN_CCPPF[key][index_big ])/2
        return CCPPF

    def kN_CCPPF_estimate(self,x_train,y_train):
        '''
        calculate Class conditional probility on training set
        '''
        x_axis = np.linspace(-3.09,3.19,500)
        labelname = self.class_name
        x_train_one = x_train[:,0] #第一个特征
        #将数据集升序排序
        unsort_xy = zip(x_train_one,y_train)
        sort_xy = sorted(unsort_xy)
        sort_xy = np.array(sort_xy)
        x_train_one,y_train = sort_xy[:,0],sort_xy[:,1]
        for key in labelname:
            index = np.argwhere(y_train==float(key))#利用标签获取该类的索引
            class_data = x_train_one[index]#利用该类的索引获取数据
            N = class_data.size
            kN = self.get_kN(N)
            #确定下界和上界
            for x_item in class_data:
                x_index = np.argwhere(class_data == x_item)[0][0]
                left_num = class_data[:x_index].size #x左边存在的样本数
                right_num = class_data[x_index:].size#x有边的样本数
                if left_num < kN:
                    lower_bound = min(class_data)#最左侧区间起点
                    upper_bound = class_data[x_index+kN-left_num]
                elif right_num < kN:
                    upper_bound = max(class_data)#最右侧区间起点
                    lower_bound = class_data[x_index-kN+right_num]
                else:
                    upper_bound =class_data[x_index+kN//2]#中间点
                    lower_bound = class_data[x_index-kN//2]
                
                update_index = np.argwhere((x_axis>lower_bound) & (x_axis<upper_bound))-1
                # self.kN_CCPPF[key][update_index] +=1.0/N
                self.kN_CCPPF[key][update_index] = kN*1.0/(N*(upper_bound - lower_bound))

    def get_kN(self,N):
        '''
        kN is a variable with respect to the number of samples N
        supposed this mapping relationship is sqrt()
        '''
        kN = int(math.sqrt(1.0*N))
        # kN = 10
        return kN

    def minimize_error_decision(self,x,posp):
        '''
        makes a decision on unknown-class sample by minimize error decision method
        '''
        # posp_dict=self.posterior_distribution
        posp_dict = posp
        decision_result = max(posp_dict,key=posp_dict.get) #取后验概率最大者
        return decision_result

    def minimize_risk_decision(self,x,posp,risk):
        '''
        makes a decision on unknown-class sample by minimize risk decision method
        '''
        risk_dict = self.calculate_risk(x,posp,risk)
        decision_result = min(risk_dict,key=risk_dict.get)
        return decision_result

    def calculate_risk(self,x,posp,risk):
        '''
        calculate the conditional risk for minimize_risk_decision
        '''
        risk_dict = dict()
        for key in self.class_name:
            risk_dict[key] = 0
        for i ,key in enumerate(self.class_name):
            for j ,prob_key in enumerate(self.class_name):
                risk_dict[key] += posp[prob_key]*risk[i][j]
        self.risk_list.append(risk_dict)
        return risk_dict

    def viz_risk(self,x_data):
        print("visualizing the conditional risk map!")
        risk_class = {}
        for key in self.class_name:
            risk_class[key] =[]
        for risk in self.risk_list:
            for key in self.class_name:
                risk_class[key].append(risk[key])
        plt.figure()
        plt.plot(x_data,risk_class[self.class_name[0]],label='class:{}'.format(self.class_name[0]))
        plt.plot(x_data,risk_class[self.class_name[1]],color='red',label='class:{}'.format(self.class_name[1]))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("条件风险分布曲线")
        plt.xlabel("x")#x轴上的名字
        plt.ylabel("R(w1|x)")#y轴上的名字
        plt.legend(loc = 'upper right')
        plt.show()
        return 0

    def viz_CCPPF(self,x_train,y_train,h1):
        x_data = np.linspace(-3.09,3.19,500)#从(-1,1)均匀取50个点
        y_data = dict()
        print("visualizing the CCPPF!")
        for key in self.class_name:
            y_data[key] = list()
            CCPPF = dict()
        if(self.CCP_estimate_method == "parzen-window"):
            for x in tqdm(x_data):
                CCPPF = self.parzen_window_estimate(x,x_train,y_train,h1)
                for key in self.class_name:
                    y_data[key].append(CCPPF[key])
        elif(self.CCP_estimate_method == "max-likelihood"):
            h1 = "None"
            for x in tqdm(x_data):
                for key in self.class_name:
                    mu,sigma = self.params_dict[key]
                    CCPPF[key] = self.Gaussian_distribution(x,mu,sigma)
                    y_data[key].append(CCPPF[key])
        elif(self.CCP_estimate_method == "k-neighbours"):
            h1 = "None"
            y_data = self.kN_CCPPF

                    
        plt.figure()
        plt.plot(x_data,y_data[self.class_name[0]],label='class:{}'.format(self.class_name[0]))
        plt.plot(x_data,y_data[self.class_name[1]],color='red',label='class:{}'.format(self.class_name[1]))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("{}估计类条件概率密度 h1={}".format(self.CCP_estimate_method,h1))
        plt.xlabel("x")#x轴上的名字
        plt.ylabel("p(x|w)")#y轴上的名字
        plt.legend(loc = 'upper right')
        plt.show()

    def predict(self,x,x_train=None,y_train=None,risk=None,h1=0.5):
        CCPPF = self.class_conditional_probability(x,x_train,y_train,h1)
        posp = self.posterior_probability(x,CCPPF,self.prior_distribution)
        if self.decision_method == "MSD":
            result = float(self.minimize_error_decision(x,posp))
        elif self.decision_method == "MRD":
            result = float(self.minimize_risk_decision(x,posp,risk))
        # print("prior_distribution:{}".format(self.prior_distribution))
        # print("class_conditional_probability:{}".format(CCPPF))
        # print("posterior_probability:{}".format(posp))
        return result

    # def multi_predict(self,x_tensor,x_train,y_train,h1=0.5):
    #     print("prediction starts!")
    #     result = list()
    #     for x in tqdm(x_tensor):
    #         CCPPF = self.class_conditional_probability(x,x_train,y_train,h1)
    #         posp = self.posterior_probability(x,CCPPF,self.prior_distribution)
    #         result.append(float(self.minimize_error_decision(x,posp)))
    #         # print("prior_distribution:{}".format(self.prior_distribution))
    #         # print("class_conditional_probability:{}".format(CCPPF))
    #         # print("posterior_probability:{}".format(posp))
    #     return result

    def multi_predict(self,x_tensor,x_train=None,y_train=None,risk=None,h1=0.5):
        print("prediction starts!")
        result = list()
        for x in tqdm(x_tensor):
            result.append(self.predict(x,x_train,y_train,risk,h1))
        return result

    def evaluate_with_new_valset(self,x_test,y_test,x_train,y_train,h1):
        y_pred = []
        print("evalutation starts!")
        for i in tqdm(range(x_test.size)):
            y_pred.append(self.predict(x_test[i][0],x_train,y_train,h1))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        return acc,prec
    
    def evaluate(self,y_pred,y_test):
        print("evalutation starts!")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        return acc,prec


if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion/Exp1-2'
    datasets_dir = os.path.join(Root_dir,"datasets")
    os.chdir(Root_dir)
    dataset_path = os.path.join(datasets_dir,'banana.dat')

    # BC = BayesianClassifier("MSD","max-likelihood")
    # BC = BayesianClassifier("MSD","parzen-window")
    BC = BayesianClassifier("MSD","k-neighbours")
    

    features,labels = BC.data_generator(dataset_path) #load data
    # features1 = np.random.normal(0,0.5,size=(2000,1))
    # features2 = np.random.normal(1,0.5,size=(2000,1))
    # features = np.hstack((features1,features2))
    # labels = np.ones((2000,))
    x_train,x_test, y_train, y_test = BC.dataset_split(features,labels,test_size = 0.2)#split data

    BC.prior_distribution = BC.prior_probility(features,labels) # calculate prior distribution
    print("prior_distribution:{}".format(BC.prior_distribution))

    y_pred = BC.multi_predict(x_test[:,0],x_train,y_train,16)
    acc,prec = BC.evaluate(y_pred,y_test)
    # acc,prec = BC.evaluate_with_new_valset(x_test,y_test,x_train,y_train,h1 = 16)#evalutation
    print("acc: {} prec:{}".format(acc,prec))

    BC.viz_CCPPF(x_train,y_train,h1 = 32)#visualizing

 
