import sys
#sys.path.append('../')

#features_trainData = '../data/2/lvae_train.npy'
#features_testData = '../data/2/lvae_test.npy'
#lableY_trainData = '../data/2/y_train.npy'
#lableY_testData = '../data/2/y_test.npy'

import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
#import pylab as pl
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib               # MatPlotLib is for making plots & figures
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition


def confusion_mtx(y_test, y_pred):
    np.set_printoptions(precision=2)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fn)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc_mtx = [accuracy, precision, recall, fscore, fpr, fnr]
    return acc_mtx

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

# Logistice Reg
def logisticReg(X_train, X_test, y_train, y_test):
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    yhat_LR = LR.predict(X_test)
    yhat_prob_LR = LR.predict_proba(X_test)
    jaccard_score(y_test, yhat_LR,pos_label=0)
    # Compute confusion matrix
    acc_mtx_LR=confusion_mtx(y_test, yhat_LR)
    return acc_mtx_LR

# SVM
def SVM_class(X_train, X_test, y_train, y_test):
    regularization = [0.01, 0.1, 10, 100]
    svm_clf4 = SVC(kernel="linear", C=regularization[3], gamma='scale')
    svm_clf4.fit(X_train, y_train)
    yhat4_svm = svm_clf4.predict(X_test)
    # Compute confusion matrix
    acc_mtx_svm = confusion_mtx(y_test, yhat4_svm)
    return acc_mtx_svm

#Random Forest
def RandomForest_class(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    yhat_randomForest = clf.predict(X_test)
    acc_mtx_randomForest = confusion_mtx(y_test, yhat_randomForest)
    return acc_mtx_randomForest

# Naive Bayes
def NB_class(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    y_pred_NB = gnb.fit(X_train, y_train).predict(X_test)
    acc_mtx_NB = confusion_mtx(y_test, y_pred_NB)
    return acc_mtx_NB

# Multi-layer Perceptron classifier
def MLP_class(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    #clf.predict_proba(X_test)
    yhat_MLP = clf.predict(X_test)
    #clf.score(X_test, y_test)
    acc_mtx_MLP = confusion_mtx(y_test, yhat_MLP)
    return acc_mtx_MLP

X_train= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/lvae_train.npy')
y_train= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/y_train.npy')
X_test= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/lvae_test.npy')
y_test= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/y_test.npy')
'''
X_trainSc,X_testSc = scale_data(X_train, X_test)
#print(X_train,X_test)

LogReg_cls=logisticReg(X_trainSc, X_testSc, y_train, y_test)
print('LogReg is running',LogReg_cls)
SVM_cls=SVM_class(X_train, X_test, y_train, y_test)
print('SVM_cls is running',SVM_cls)
RandomForest_cls=RandomForest_class(X_trainSc, X_testSc, y_train, y_test)
print('RandomForest_cls is running',RandomForest_cls)
NB_cls=NB_class(X_trainSc, X_testSc, y_train, y_test)
print('NB_cls is running',NB_cls)
MLP_cls=MLP_class(X_trainSc, X_testSc, y_train, y_test)
print('MLP_cls is running',MLP_cls)


Columns_accuracy = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'FNR']
df_table_summary_LR = pd.DataFrame([LogReg_cls], columns=Columns_accuracy, index=['Logistic Reg']).round(2)
df_table_summary_SVM = pd.DataFrame([SVM_cls], columns=Columns_accuracy, index=['SVM']).round(2)
df_table_summary_RandomForest = pd.DataFrame([RandomForest_cls], columns=Columns_accuracy, index=['Random Forest']).round(2)
df_table_summary_NB = pd.DataFrame([NB_cls], columns=Columns_accuracy, index=['Naive Bayes']).round(2)
df_table_summary_MLP = pd.DataFrame([MLP_cls], columns=Columns_accuracy, index=['MLP']).round(2)
df_all = pd.concat([df_table_summary_LR, df_table_summary_SVM, df_table_summary_RandomForest, df_table_summary_NB, df_table_summary_MLP])
print(df_all)

LogReg_cls32=logisticReg(X_trainSc[:,0:32], X_testSc[:,0:32], y_train, y_test)
SVM_cls32=SVM_class(X_train[:,0:32], X_test[:,0:32], y_train, y_test)
RandomForest_cls32=RandomForest_class(X_trainSc[:,0:32], X_testSc[:,0:32], y_train, y_test)
NB_cls32=NB_class(X_trainSc[:,0:32], X_testSc[:,0:32], y_train, y_test)
MLP_cls32=MLP_class(X_trainSc[:,0:32], X_testSc[:,0:32], y_train, y_test)

df_table_summary_LR32 = pd.DataFrame([LogReg_cls32], columns=Columns_accuracy, index=['Logistic Reg']).round(2)
df_table_summary_SVM32 = pd.DataFrame([SVM_cls32], columns=Columns_accuracy, index=['SVM']).round(2)
df_table_summary_RandomForest32 = pd.DataFrame([RandomForest_cls32], columns=Columns_accuracy, index=['Random Forest']).round(2)
df_table_summary_NB32 = pd.DataFrame([NB_cls32], columns=Columns_accuracy, index=['Naive Bayes']).round(2)
df_table_summary_MLP32 = pd.DataFrame([MLP_cls32], columns=Columns_accuracy, index=['MLP']).round(2)
df_all32 = pd.concat([df_table_summary_LR32, df_table_summary_SVM32, df_table_summary_RandomForest32, df_table_summary_NB32, df_table_summary_MLP32])
print('     ')
print('Results for the first 32 col:')
print(df_all32)


LogReg_cls10=logisticReg(X_trainSc[:,32:42], X_testSc[:,32:42], y_train, y_test)
SVM_cls10=SVM_class(X_train[:,32:42], X_test[:,32:42], y_train, y_test)
RandomForest_cls10=RandomForest_class(X_trainSc[:,32:42], X_testSc[:,32:42], y_train, y_test)
NB_cls10=NB_class(X_trainSc[:,32:42], X_testSc[:,32:42], y_train, y_test)
MLP_cls10=MLP_class(X_trainSc[:,32:42], X_testSc[:,32:42], y_train, y_test)

df_table_summary_LR10 = pd.DataFrame([LogReg_cls10], columns=Columns_accuracy, index=['Logistic Reg']).round(2)
df_table_summary_SVM10 = pd.DataFrame([SVM_cls10], columns=Columns_accuracy, index=['SVM']).round(2)
df_table_summary_RandomForest10 = pd.DataFrame([RandomForest_cls10], columns=Columns_accuracy, index=['Random Forest']).round(2)
df_table_summary_NB10 = pd.DataFrame([NB_cls10], columns=Columns_accuracy, index=['Naive Bayes']).round(2)
df_table_summary_MLP10 = pd.DataFrame([MLP_cls10], columns=Columns_accuracy, index=['MLP']).round(2)
df_all10 = pd.concat([df_table_summary_LR10, df_table_summary_SVM10, df_table_summary_RandomForest10, df_table_summary_NB10, df_table_summary_MLP10])
print('     ')
print('Results for the second 10 col:')
print(df_all10)
'''

def normalize_data_min_max_sk(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

#PCA
def PCA_featureSel(X_data):
    print('wait ... running PCA feature selection')
    X=normalize_data_min_max_sk(X_data)
    pca = decomposition.PCA(n_components=None).fit(X)
    Y=pca.transform(X)
    return Y

#X_train= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/lvae_train.npy')
pca_x=PCA_featureSel(X_train)
print(pca_x)

'''
fig, ax = plt.subplots()
ax.scatter(pca_x[:,0], pca_x[:,1], c=y_train, cmap='coolwarm')
ax.set(xlabel='PCA1', ylabel='PCA2')
#x1_min, x1_max = xdat[:,0].min() - .5, xdat[:,0].max() + .5
#x2_min, x2_max = xdat[:,1].min() - .5, xdat[:,1].max() + .5
#xx1 = np.linspace(x1_min, x1_max)
#ax.plot(xx1, xx2, 'k')
#ax.scatter(sv[:,0], sv[:,1], s=100, facecolors='none', edgecolors='k')'''


#RandomForestClassifier Feature Selection
def RandomForest_FeatureSel(X_train,y_train,n):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    aa=clf.feature_importances_
    ranked = np.argsort(aa)
    selected_indices = ranked[::-1][:n]
    return X_train[:,selected_indices]

#print(RandomForest_FeatureSel(X_train,len(X_train)))

root='C:/Users/Owner/PycharmProjects/Big-Data/data/new_features'

y_train= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/train_label.npy')
y_test= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/test_label.npy')
print(len(y_train),len(y_test))

X_train_vae= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/vae_train.npy')
X_test_vae= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/vae_test.npy')
print(len(X_train_vae),len(X_test_vae))


X_train_lda= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/lda_tf_train.npy')
X_test_lda= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/lda_tf_test.npy')
print(len(X_train_lda),len(X_test_lda))

X_train_vae_lda= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/lvae_train_vae_lda_tf.npy')
X_test_vae_lda= np.load(root+'/ISOT_ep_30_seq_len_45_topics_64_latent_dim_64/features/lvae_test_vae_lda_tf.npy')
print(len(X_train_vae_lda),len(X_test_vae_lda))



X_train_vaeRD=RandomForest_FeatureSel(X_train_vae,y_train,32)
X_test_vaeRD=RandomForest_FeatureSel(X_test_vae,y_test,32)


X_train_ldaRD=RandomForest_FeatureSel(X_train_lda,y_train,32)
X_test_ldaRD=RandomForest_FeatureSel(X_test_lda,y_test,32)