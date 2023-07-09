# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:44:27 2023

@author: qiubinxu
"""

import streamlit as st
import pandas as pd #处理数据所用库
import numpy as np
import requests #工具，访问服务器
import numpy as np#加载数据所用库
import pandas as pd #处理数据所用库
import xgboost
import xgboost as xgb#极限梯度提升机所用库
from xgboost import XGBClassifier#分类算法#加载极限梯度提升机中分类算法
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree #导入需要的模块
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split#模型选择将数据集分为测试集和训练集
from sklearn.metrics import accuracy_score#模型最终的预测准确度分数
from sklearn.datasets import load_iris#加载鸢尾花集数据
from sklearn.datasets import load_boston#加载鸢尾花集数据
from sklearn.datasets import load_breast_cancer#加载鸢尾花集数据
import matplotlib#加载绘图工具
from xgboost import plot_importance#加载极限梯度提升机中重要性排序函数
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score   # 准确率
import scipy.stats as stats
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as CVS
from xgboost import plot_importance
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from IPython.display import display
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
import eli5
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from eli5.sklearn import PermutationImportance
from IPython.display import display, Image
from sklearn.model_selection import cross_val_score,StratifiedKFold
st.title('Application of Machine Learning Methods to Predict Distant Metastasis in Rectal Cancer Patients'.center(33, '-'))
classes = {0:'NDM',1:'DM'}
st.sidebar.expander('')     # expander必须接受一个 label参数，我这里留了一个空白
st.sidebar.subheader('Variable')       # 副标题
# st.selectbox:创造一个下拉选择框的单选题，接收参数: (题目名称， 题目选项)
Age = st.sidebar.selectbox('Age', ['≤50','>50'])
Age_map = 1 if Age == '>50' else 0
Grade=st.sidebar.selectbox('Grade', ['Grade I','Grade II','Grade III','Grade IV'])
Grade_map = {'Grade I':0,'Grade II':1,'Grade III':2,'Grade IV':3}
T_stage = st.sidebar.selectbox('T_stage',["T1","T2","T3","T4"])# 选择聚类中心
T_stage_map = {'T1':1,'T2':2,'T3':3,'T4':4}
N_stage=st.sidebar.selectbox('N_stage',['N0','N1','N2'])
N_stage_map = {'N0':1,'N1':2,'N2':3}
CEA=st.sidebar.selectbox('CEA', ['Negative','Borderline','Positive','Unknown'])
CEA_map = {'Negative':0,'Borderline':1,'Positive':2,'Unknown':3}
Perineural_Invasion=st.sidebar.selectbox('Perineural_Invasion', ['No','Yes','Unknown'])
Perineural_Invasion_map = {'No':0,'Yes':1,'Unknown':2}
Tumor_Deposits=st.sidebar.selectbox('Tumor_Deposits', ['No','Yes','Unknown'])
Tumor_Deposits_map = {'No':0,'Yes':1,'Unknown':2}
Tumor_size = st.sidebar.selectbox('Tumor.size', ['≤5','>5'])
Tumor_size_map = 1 if Tumor_size == '>5' else 0
Radiation = st.sidebar.selectbox('Radiation', ['No','Yes'])
Radiation_map = 1 if Radiation == 'Yes' else 0
Chemotherapy = st.sidebar.selectbox('Chemotherapy', ['No','Yes'])
Chemotherapy_map = 1 if Chemotherapy == 'Yes' else 0
filename = 'model3.txt'
x = []
x.extend([Age_map,Grade_map[Grade],T_stage_map[T_stage],N_stage_map[N_stage],
          CEA_map[CEA],Perineural_Invasion_map[Perineural_Invasion],
         Tumor_Deposits_map[Tumor_Deposits],Tumor_size_map,
         Radiation_map,Chemotherapy_map])
x=np.array(x).reshape(1,10)
import pickle
if st.button("Predict"):
    #  predict_class()
    import os
    if os.path.exists(filename):
        with open(filename, 'rb') as fq:
            modelRF = pickle.load(fq, encoding='bytes')
            y_pred = modelRF.predict_proba(x)
            print(max(y_pred[:,1]))
            st.header('Probability of distant metastasis: %.2f %%' % (max(y_pred[:,1])* 100))