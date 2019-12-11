"""
A simple implementation of Bayesian Classifier
@data: 2019.12.9
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
    def __init__(self,data_path,decision_method = 'MSD',CCP_estimate_method="max-likelihood"):
        self.data_path = data_path
        self.class_name = [] #类名
        self.decision_method = decision_method
        self.CCP_estimate_method = CCP_estimate_method
        self.features,self.labels = self.data_generator(data_path)#生成generator
        self.x_train,self.x_test, self.y_train, self.y_test = self.dataset_split(self.features,self.labels)#划分数据集
        self.prior_distribution = self.prior_probility(self.x_train, self.y_train)
        self.params_dict = {} #存放各类概率密度函数的参数 {'classname',(mu,std)}
        self.CCPPF = {}  #类条件概率密度函数 {'classname',p(x|w)}
        self.posterior_distribution = {} 
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def data_generator(self,data_path):
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
        return np.array(features),np.array(labels)

    def dataset_split(self,train_data,train_target,test_size = 0.3,random_state=None):
        return train_test_split(train_data,train_target,test_size=test_size, random_state=random_state)
        
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
    
    def class_conditional_probability(self,x,h1=0.5):
        '''
        computed class_conditional_probability p(x|w)
        '''
        if self.CCP_estimate_method == "max-likelihood":
            if not self.params_dict:
                self.params_dict = self.maximum_likelihood_estimate(self.x_train,self.y_train)
            labelname = self.class_name
            CCPPF = {}
            for key in labelname:
                mu,sigma = self.params_dict[key]
                CCPPF[key] = self.Gaussian_distribution(x,mu,sigma)
        if self.CCP_estimate_method == "parzen-window":
            CCPPF = self.parzen_window_estimate(x,self.x_train,self.y_train,h1)

        # self.CCPPF = CCPPF
        return CCPPF

    def posterior_probability(self,x,CCPPF,prior):
        '''
        computed posterior_distribution p(w|x)
        '''
        labelname = self.class_name
        self.posterior_distribution ={}
        for key in labelname:
            #Bayes formula : p(w|x) =^ p(x|w)*p(w)
            self.posterior_distribution[key] = prior[key]*CCPPF[key]
        
        return self.posterior_distribution
    
    def Gaussian_distribution(self,x,mu,sigma):
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
        '''
        labelname = self.class_name
        x_train_one = x_train[:,0] #第一个特征
        window_h = self.get_parzenwindow_hN(h1)
        CCPPF = {}
        # print("start calculate CCPPF!\n")
        for key in labelname:
            index = np.argwhere(y_train==float(key))#利用标签获取该类的索引
            class_data = x_train_one[index]#利用该类的索引获取数据
            N = len(class_data)
            CCP = 0
            for i in range(N):
                # CCP += self.Gaussian_distribution(x,class_data[i],math.sqrt(window_h))*1.0/(N*window_h) #计算Pn(x)
                CCP += self.Gaussian_distribution(x,class_data[i],window_h)*1.0/(N*window_h) #计算Pn(x)
            CCPPF[key] = CCP
        return CCPPF

    # def parzen_window_function(self,x,class_data,window_h,method = 'GD'):
    #     if method == 'GD':
    #         return self.Gaussian_distribution(x,class_data[i],math.sqrt(window_h))*1.0/(N*window_h)

    def get_parzenwindow_hN(self,h1):
        '''
        calculated parzen window's hN ,which is the width of the window
        via VN = hN = h1/sqrt(N)
        '''
        return h1/math.sqrt(1.0*len(self.x_train)) 

    def minimize_error_decision(self,x,posp):
        # posp_dict=self.posterior_distribution
        posp_dict = posp
        decision_result = max(posp_dict,key=posp_dict.get) #取后验概率最大者
        return decision_result

    def viz_CCPPF(self):
        x_data = np.linspace(-3.09,3.19,250)#从(-1,1)均匀取50个点
        y_data = dict()
        h1 = 64
        for key in self.class_name:
            y_data[key] = list()
        print("visualizing the CCPPF!")
        for x in tqdm(x_data):
            CCPPF = self.parzen_window_estimate(x,self.x_train,self.y_train,h1)
            for key in self.class_name:
                y_data[key].append(CCPPF[key])
        
        # plt.figure()
        # # figure splits into 2 rows, 1 col, plot to the 1st sub-fig
        # # plt.subplot(121)
        # plt.plot(x_data,y_data[self.class_name[0]])
        # plt.xlabel("x")#x轴上的名字
        # plt.ylabel("p(x|w)")#y轴上的名字
        plt.figure()
        plt.plot(x_data,y_data[self.class_name[0]],label='class:{}'.format(self.class_name[0]))
        plt.plot(x_data,y_data[self.class_name[1]],color='red',label='class:{}'.format(self.class_name[1]))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("Parzen窗口估计类条件概率密度 h1={}".format(h1))
        plt.xlabel("x")#x轴上的名字
        plt.ylabel("p(x|w)")#y轴上的名字
        plt.legend(loc = 'upper right')
        plt.show()

    def predict(self,x):
        CCPPF = self.class_conditional_probability(x,64)
        posp = self.posterior_probability(x,CCPPF,self.prior_distribution)
        result = float(self.minimize_error_decision(x,posp))
        # print("prior_distribution:{}".format(self.prior_distribution))
        # print("class_conditional_probability:{}".format(CCPPF))
        # print("posterior_probability:{}".format(posp))
        return result
    
    def multi_predict(self,x_tensor):
        result = list()
        for x in x_tensor:
            CCPPF = self.class_conditional_probability(x)
            posp = self.posterior_probability(x,CCPPF,self.prior_distribution)
            result.append(float(self.minimize_error_decision(x,posp)))
            # print("prior_distribution:{}".format(self.prior_distribution))
            # print("class_conditional_probability:{}".format(CCPPF))
            # print("posterior_probability:{}".format(posp))
        return result

    def evaluate(self):
        x_test,y_test = self.x_test,self.y_test
        y_pred = []
        print("evalutation starts!")
        for i in tqdm(range(len(x_test))):
            y_pred.append(self.predict(x_test[i][0]))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        print("acc: {} prec:{}".format(acc,prec))
        return acc,prec


if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion'
    os.chdir(Root_dir)
    dataset_path = os.path.join(Root_dir,'./banana.dat')
    BC = BayesianClassifier(dataset_path,CCP_estimate_method = 'parzen-window')
    acc,prec = BC.evaluate()
    BC.viz_CCPPF()

 
