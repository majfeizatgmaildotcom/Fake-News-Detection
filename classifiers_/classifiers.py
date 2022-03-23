import sys
#sys.path.append('../')
features_trainData = '../data/2/lvae_train.npy'
features_testData = '../data/2/lvae_test.npy'
lableY_trainData = '../data/2/y_train.npy'
lableY_testData = '../data/2/y_test.npy'
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
#import pylab as pl
import numpy as np
#import scipy.optimize as opt
#from sklearn import preproces
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#X = np.loadtxt('xvals.dat')
#y = np.loadtxt('yvals.dat')

#X = np.load('vae_fnd.npy')#[:1:5]
#y= np.load('test_label.npy')[:,1]

X_train= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/lvae_train.npy')
y_train= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/y_train.npy')
X_test= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/lvae_test.npy')
y_test= np.load('C:/Users/Owner/PycharmProjects/Big-Data/data/2/y_test.npy')

def confusion_mtx(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fn)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc_mtx = [accuracy, precision, recall, fscore, fpr, fnr]
    return acc_mtx



from sklearn.svm import SVC


#print(X_test) #, y_train,
#print(y_test)

#print(X_test[:,0:4]) #, y_train,
#print(y_test[0:4])

# Logistice Reg
def logisticReg(X_train, X_test, y_train, y_test):
    from sklearn import preprocessing
    from sklearn.linear_model import LogisticRegression
    X = preprocessing.StandardScaler().fit(X_train).transform(X_train)
    from sklearn.metrics import confusion_matrix
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    yhat_LR = LR.predict(X_test)
    yhat_prob_LR = LR.predict_proba(X_test)
    from sklearn.metrics import jaccard_score
    jaccard_score(y_test, yhat_LR,pos_label=0)
    from sklearn.metrics import classification_report, confusion_matrix


    # Compute confusion matrix
    cnf_matrix_LR = confusion_matrix(y_test, yhat_LR, labels=[1,0])
    np.set_printoptions(precision=2)
    #print(cnf_matrix_LR[0,0])
    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix_LR, classes=['class=1 Fake','class=0 Real'],normalize= False,  title='Confusion matrix')

    #print (classification_report(y_test, yhat_LR))

    Model_accuracy_logistic = accuracy_score(y_test, yhat_LR)
    Model_accuracy_logistic



    from sklearn.metrics import log_loss
    log_loss(y_test, yhat_prob_LR)


    LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
    yhat_prob2_LR = LR2.predict_proba(X_test)
    #print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2_LR))


    TP_LR=cnf_matrix_LR[0,0]
    FN_LR=cnf_matrix_LR[0,1]
    FP_LR=cnf_matrix_LR[1,0]
    TN_LR=cnf_matrix_LR[1,1]
    Accurancy_LR=(TP_LR+TN_LR)/(TP_LR+TN_LR+FP_LR+FN_LR)
    Precision_LR=TP_LR/(TP_LR+FN_LR)
    Recall_LR=TP_LR/(TP_LR+FN_LR)
    F1_Score_LR=(2*Precision_LR*Recall_LR)/(Precision_LR+Recall_LR)
    FPR_LR=FP_LR/(FP_LR+TN_LR)
    FNR_LR=FN_LR/(FN_LR+TP_LR)
    Matrix_acc_LR=np.array([TP_LR,TN_LR,FN_LR,FP_LR,Accurancy_LR,Precision_LR,Recall_LR,F1_Score_LR,FPR_LR,FNR_LR])
    Columns_accurancy=['TP', 'TN', 'FN', 'FP', 'Accurancy','Precision', 'Recall','F1-Score','FPR','FNR']
    #Matrix_data = np.array([regularization,accuracy_scores]).T
    df_table_summary_LR= pd.DataFrame([Matrix_acc_LR], columns=Columns_accurancy ,index=['Logestic Reg']).round(2)
    #display(df_table_summary_LR)
    #df_table_summary_LR

    # In[52]:
    return df_table_summary_LR

#acc_LRR=logestic(X_train, X_test, y_train, y_test)
#print(acc_LRR)

#SVM
def SVM_class(X_train, X_test, y_train, y_test):
    regularization = [0.01, 0.1, 10, 100]
    from sklearn.svm import SVC

    '''svm_clf1 = SVC(kernel="linear", C=regularization[0], gamma='scale') 
    svm_clf1.fit(X_train, y_train)
    yhat1_svm = svm_clf1.predict(X_test)
    #print(yhat_svm)
    
    svm_clf2 = SVC(kernel="linear", C=regularization[1], gamma='scale') 
    svm_clf2.fit(X_train, y_train)
    yhat2_svm = svm_clf2.predict(X_test)
    
    svm_clf3 = SVC(kernel="linear", C=regularization[2], gamma='scale') 
    svm_clf3.fit(X_train, y_train)
    yhat3_svm = svm_clf3.predict(X_test)
    
    svm_clf4 = SVC(kernel="linear", C=regularization[3], gamma='scale') 
    svm_clf4.fit(X_train, y_train)
    yhat4_svm = svm_clf4.predict(X_test)
    
    Model1_accuracy = accuracy_score(y_test, yhat1_svm)
    Model2_accuracy = accuracy_score(y_test, yhat2_svm)
    Model3_accuracy = accuracy_score(y_test, yhat3_svm)
    Model4_accuracy = accuracy_score(y_test, yhat4_svm)
    accuracy_scores = [Model1_accuracy, Model2_accuracy, Model3_accuracy, Model4_accuracy]
    
    
    df_svm_acc= pd.DataFrame(np.array([regularization,accuracy_scores]).T ,  columns=['Regularization','Accuracy Scores'],index=[1,2,3,4])
    display(df_svm_acc)'''


    # In[53]:

    svm_clf4 = SVC(kernel="linear", C=regularization[3], gamma='scale')
    svm_clf4.fit(X_train, y_train)
    yhat4_svm = svm_clf4.predict(X_test)
    #print(yhat4_svm)
    # Compute confusion matrix
    cnf_matrix_svm = confusion_matrix(y_test, yhat4_svm, labels=[1,0])
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix_svm, classes=['class=1 Fake','class=0 Real'],normalize= False,  title='Confusion matrix')
    # In[54]:
    #print (classification_report(y_test, yhat4_svm))
    # In[55]:
    TP_svm=cnf_matrix_svm[0,0]
    FN_svm=cnf_matrix_svm[0,1]
    FP_svm=cnf_matrix_svm[1,0]
    TN_svm=cnf_matrix_svm[1,1]
    Accurancy_svm=(TP_svm+TN_svm)/(TP_svm+TN_svm+FP_svm+FN_svm)
    Precision_svm=TP_svm/(TP_svm+FN_svm)
    Recall_svm=TP_svm/(TP_svm+FN_svm)
    F1_Score_svm=(2*Precision_svm*Recall_svm)/(Precision_svm+Recall_svm)
    FPR_svm=FP_svm/(FP_svm+TN_svm)
    FNR_svm=FN_svm/(FN_svm+TP_svm)
    Matrix_acc_svm=np.array([TP_svm,TN_svm,FN_svm,FP_svm,Accurancy_svm,Precision_svm,Recall_svm,F1_Score_svm,FPR_svm,FNR_svm])
    Columns_accurancy=['TP', 'TN', 'FN', 'FP', 'Accurancy','Precision', 'Recall','F1-Score','FPR','FNR']
    df_table_svm = pd.DataFrame([Matrix_acc_svm], columns=Columns_accurancy ,index=['SVM']).round(2)
    return df_table_svm

LogReg_cls=logisticReg(X_train, X_test, y_train, y_test)
SVM_cls=SVM_class(X_train, X_test, y_train, y_test)
df=pd.concat([LogReg_cls,SVM_cls])
#print(df)
print(df.T)


LogReg_cls32=logisticReg(X_train[:,0:32], X_test[:,0:32], y_train, y_test)
SVM_cls32=SVM_class(X_train[:,0:32], X_test[:,0:32], y_train, y_test)
print('     ')
print('Results for the first 32 col:')
df32=pd.concat([LogReg_cls32,SVM_cls32])
print(df32.T)


LogReg_cls10=logisticReg(X_train[:,32:42], X_test[:,32:42], y_train, y_test)
SVM_cls10=SVM_class(X_train[:,32:42], X_test[:,32:42], y_train, y_test)
print('     ')
print('Results for the second 10 col:')
df10=pd.concat([LogReg_cls10,SVM_cls10])
print(df10.T)


def svm_linear(X_train, X_test, y_train, y_test, regularization=100):
    svm_clf4 = SVC(kernel="linear", C=regularization, gamma='scale')
    svm_clf4.fit(X_train, y_train)
    y_pred = svm_clf4.predict(X_test)


