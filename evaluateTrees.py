# -*- coding: utf-8 -*-
__author__ = 'Xin'

from sklearn import tree
import pandas as pd
import numpy as np
from time import clock
from sklearn.ensemble import IsolationForest
from math import log
from sklearn.model_selection import StratifiedKFold as skfold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from random import shuffle
import PathWalk as pw
from matplotlib import pyplot as plt
import stat_real_final as sT
import SCiForest as scF
import os.path as path

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict

rootPath = "/media/xinxin/文档/UnionPayProject/randomForest/有效数据集/"
    #"/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/AllData/norm_UCI/Norm/"
# 下面直接返回归一化的数据了,以后的数据干脆都不要header好了
def loadData(fname, flag=1):
    data = None
    if flag == 1:
        if 'PenDigits' in fname or 'KDDCup99' in fname:
            data = pd.read_csv(fname, na_values='?')
        else:
            data = pd.read_csv(fname, na_values='?', header=None)
        if fname.find("Norm") != -1:
            classlabels = data[data.columns[-1]]
            del data[data.columns[-1]]
            return data.get_values(), classlabels
    elif fname.find(".txt") != -1:
        data = pd.read_csv(fname, header=None, na_values='?')
    else:
        data = pd.read_csv(fname, na_values="?")
    classlabels = data[data.columns[-1]]
    del data[data.columns[-1]]
    data = data.bfill().ffill()
    for c in data.columns:
        maxd = max(data[c])
        mind = min(data[c])
        z = maxd - mind
        if z > 0:
            data[c] = (data[c] - mind) * 1.0 / z
        else:
            data[c] = np.zeros(len(data))
    return data.get_values(), classlabels

def transResult(real, predict):
    categories = list(set(real))
    majority = categories[0] if real.count(categories[0]) > real.count(categories[1]) else categories[1]
    a = list(map(lambda t: 0 if t == majority else 1, real))
    b = predict
    if set(predict) != set([0, 1]):
        b = list(map(lambda t: 0 if t == 1 else 1, predict))
    return [a, b]  #前面是真实的，后面是预测的

def formula_auc(rr):
    categories = list(set(rr[0]))
    majority = categories[0] if rr[0].count(categories[0]) > rr[0].count(categories[1]) else categories[1]
    real = list(map(lambda t: 0 if t == majority else 1, rr[0]))
    xdata = list(zip(real,rr[1]))
    xdata = sorted(xdata,key=lambda t:t[1])
    sumYi = 0
    for i,j in enumerate(xdata):
        if xdata[0] == 1:
            sumYi += i
    N = rr[0].count(majority)
    return (sumYi - (len(rr[0])-N)*(len(rr[0])-N+1)/2)/(len(rr[0])-N)/N

def messure(result):
    # result = transResult(result[0], result[1])

    fpr, tpr, thresholds = metrics.roc_curve(result[0], result[1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    err = [[0, 0], [0, 0]]
    s = zip(result[0], result[1])
    for i in s:
        err[int(i[0])][int(i[1])] += 1
    PR = float(err[0][0] + err[1][1]) / (err[0][0] + err[1][1] + err[0][1] + err[1][0])
    # RR1 = err[1][1]/float(err[1][0]+err[1][1]) #if err[1][0]+err[1][1]>0 else 0
    # PR1 = err[1][1]/float(err[0][1]+err[1][1])  if err[0][1]+err[1][1]>0 else 0
    # FM1 = 2*PR1*RR1/(PR1+RR1) if PR1+RR1>0 else 0
    # resultstr = "\t".join([str(err),str(PR),str(PR1),str(RR1),str(FM1),str(roc_auc)])
    resultStrShort = "\t".join([str(err), str(PR), str(roc_auc)])
    #resultstr = "\t".join(["confusion matric:",str(err),"PR\tPR1\tRR1\tFM1",str(PR),str(PR1),str(RR1),str(FM1)])+"\n"
    return resultStrShort

    print ("confusion matric:\n", err)
    #print("PR: %f, PR1: %f, RR1: %f, FM1: %f" %(PR,PR1,RR1,FM1))
    print ("PR\tRR1\t", PR, RR1)
    print (set(result[1]))
    print (result[1][:13])

def avgStatValidate(data, labels, bf=1.0, flag=1,dname=''):
    result = [[], []]
    if flag == 1:
        for i in range(10):
            stree = sT.final_stat(bf=bf)
            a = stree.fit(data)
            result[0] = result[0] + list(labels)
            result[1] = result[1] + stree.classify(a, data)
        return result

    # for i in range(10):
        # 此处是非正规十折交叉
        # X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.9)
        # stree = sT.final_stat(bf=bf)
        # a = stree.fit(X_train)
        # #用test数据需要下面两行一套修改
        # result[0] = result[0] + list(y_train)
        # result[1] = result[1] + stree.classify(a, X_train)

    kFold = skfold(10)

    # f = open('/media/xinxin/文档/UnionPayProject/randomForest/分析与结果文档/result_OIT_realData.txt','a+')
    # f.write(dname)

    for trainIndex, testIndex in kFold.split(data,labels):
        # 正规的十折交叉，但是该十折交叉保证data，label都是array
        X_train, X_test = data[trainIndex],data[testIndex]
        y_test = labels[testIndex]
        stree = sT.final_stat(bf=bf)

        t0 = clock()
        a = stree.fit(X_train)
        print('time_',clock()-t0,end='\t')
        return

        result[0] = result[0] + list(y_test)
        pred = stree.classify(a, X_test)
        result[1] = result[1] + pred

        f.write('\t' + '%.3f' % metrics.roc_auc_score(1.-y_test, pred))

    # f.write('\n')
    # f.close()
    # return result

def statForestValidate(data, labels, nTrees,mode=1,ifgrad=0,dname=''):
    results = [[],[]]
    # # 原始方法
    # if mode ==1:
    #     forest = sT.statForest(data, nTrees,mode=ifgrad)  #ifgrad=0:不用梯度，1则用梯度
    # else:
    #     forest = scF.statForest(data, nTrees)
    # for i in range(10):
    #     estimaters = forest.fit(train_size= 0.3)
    #     results[0] = results[0] + list(labels)
    #     results[1] = results[1]+ list(forest.predict(estimaters, data))
    # if mode == 1:
    #     print(forest.sTree.count)

    #十折交叉,mode==1则为OIF，其他为sciforest
    kFold = skfold(10)
    if mode == 1:
        # f = open('/media/xinxin/文档/UnionPayProject/randomForest/分析与结果文档/result_OIF_realData.txt', 'a+')
        # f.write(dname)

        for trainIndex, testIndex in kFold.split(data,labels):
            # 正规的十折交叉，但是该十折交叉保证data，label都是array
            X_train, X_test = data[trainIndex], data[testIndex]
            y_test = labels[testIndex]
            forest = sT.statForest(X_train, nTrees, mode=ifgrad)

            t0 = clock()
            estimaters = forest.fit(train_size=0.3)
            print('time_', clock() - t0,end='\t')
            return

            results[0] = results[0] + list(y_test)
            pred = forest.predict(estimaters, X_test)
            results[1] = results[1] + list(pred)

        #     f.write('\t' + '%.3f' % metrics.roc_auc_score(1-y_test, pred))
        # f.write('\n')
        # f.close()
    else:
        f = open('/media/xinxin/文档/UnionPayProject/randomForest/分析与结果文档/result_SCiF_realData.txt', 'a+')
        f.write(dname)
        for trainIndex, testIndex in kFold.split(data,labels):
            X_train, X_test = data[trainIndex], data[testIndex]
            y_test = labels[testIndex]
            forest = scF.statForest(X_train, nTrees)
            estimaters = forest.fit()
            results[0] = results[0] + list(y_test)
            pred = forest.predict(estimaters, X_test)
            results[1] = results[1] + list(pred)

            f.write('\t' + '%.3f' % metrics.roc_auc_score(1-y_test, pred))
        f.write('\n')
        f.close()
    return results

def avgValidate(data, labels, clf, flag=1,algo='iForest',dname=''):
    result = [[], []]
    if flag == 1:#计算十次
        for i in range(10):
            clf.fit(data, labels)
            result[0] = result[0] + list(labels)
            result[1] = result[1] + list(1. - clf.decision_function(data))
        return result

    kFold = skfold(10)
    f = open('/media/xinxin/文档/UnionPayProject/randomForest/分析与结果文档/result_{}_realData.txt'.format(algo), 'a+')
    f.write(dname)
    for trainIndex, testIndex in kFold.split(data,labels):
        # 正规的十折交叉，但是该十折交叉保证data，label都是array
        X_train, X_test = data[trainIndex], data[testIndex]
        y_test = labels[testIndex]
        clf.fit(X_train)
        result[0] = result[0] + list(y_test)
        if algo.lower() == 'lof':
            pred = 1. - clf._decision_function(X_test)
        else:
            pred = 1. - clf.decision_function(X_test)
        result[1] = result[1] + list(pred)

        f.write('\t' + '%.3f' % metrics.roc_auc_score(y_test, pred))
    f.write('\n')
    f.close()
    return result

def createData(stdNormal, stdOutlier):

    np.random.seed(10)
    d1 = np.random.normal(6, stdNormal, size=(900, 5))
    d2 = np.random.normal(-3, stdOutlier, size=(100, 5))
    # d3 = np.random.normal(5, stdOutlier, size=(50, 5))
    gaussinData = np.vstack((d1, d2))
    for i in range(5):
        maxd = max(gaussinData[:,i])
        mind = min(gaussinData[:,i])
        z = maxd - mind
        gaussinData[:, i] = (gaussinData[:,i] - mind) * 1.0 / z
    labels = np.array([0 for i in range(900)] + [1 for i in range(100)])
    data = np.c_[gaussinData,labels]
    np.random.shuffle(data)
    # f = open("E:\\UnionPayProject\\randomForest\\data\\StatTestData\\createStd"+str(stdOutlier)+".txt","w+")
    # for i in data:
    #     line = ",".join(map(str,i))
    #     f.write(line+"\n")
    # f.close()
    # plt.plot(d1[:, 1], d1[:, 2], '.', markersize=8,label='normal')  #
    # plt.plot(d2[:, 1], d2[:, 2], '+', markersize=12,label='abnormal')  #
    # plt.xlabel('feat1 value', fontsize=24)  #
    # plt.ylabel('feat2 value', fontsize=24)  # , fontsize=36
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(fontsize=20, frameon=False, loc='upper left')  #
    # plt.title("stdIn=%d,stdOut=%d" % (stdNormal, stdOutlier), fontsize=36)
    # plt.show()
    # plt.savefig(r"C:\Users\Xin\Pictures\stdIn=%d,stdOut=%do.eps" % (stdNormal, stdOutlier))
    return data[:,:-1], data[:,-1]

def twoPointDepth():
    # dpath = '/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/'
    # data = pd.read_csv(dpath+"createStd5_100.txt", header=None)
    # del data[data.columns[-1]]
    # data = data.get_values()

    # onetest = data[120]
    # twotest = data[90]
    data, classlabels = loadData(rootPath + '/Norm/3PMNorm.csv')
    onetest = data[0]  # 异常
    twotest = data[1]  # 正常

    dataSize = len(data)

    testD = [onetest, twotest]
    for samplesize in [0.25,0.5,0.75]:
        f = open(rootPath + "/convergence/pima_depth_iForest_{}.txt".format(samplesize), "w+")
        f.write("ntrees\tavgDepth\tavgDepth2\n")
        cn1 = 2. * (np.log(samplesize*dataSize) + 0.5772156649) - 2*(samplesize*dataSize-1)/(samplesize*dataSize)
        print(cn1)

        for i in range(2, 1000):
            clf = IsolationForest(i, max_samples=samplesize)
            clf.fit(data)
            a = clf.decision_function(testD)
            b = -np.log2(0.5 - a) * cn1
            f.write(str(i) + "\t" + str(b[0]) + "\t" + str(b[1]) + "\n")
        f.close()

        # f2 = open(dpath+"createStd5_depth_stat{}.txt".format(samplesize), "w+")
        # f2.write("ntrees\tavgDepth\tavgDepth2\n")
        # for i in range(2, 1000):
        #     forest = sT.statForest(data, i)
        #     estimaters = forest.fit(train_size=samplesize)
        #     b = forest.predict(estimaters,testD)
        #     f2.write(str(i) + "\t" + str(b[0]) + "\t" + str(b[1]) + "\n")
        # f2.close()

def twoPointDepth2():
    # dpath = '/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/'
    # data = pd.read_csv(dpath + "createStd5.txt", header=None) #自创数据集的收敛性计算
    # del data[data.columns[-1]]
    # for c in data.columns:
    #     maxd = max(data[c])
    #     mind = min(data[c])
    #     z = maxd - mind
    #     if z > 0:
    #         data[c] = (data[c] - mind) * 1.0 / z
    #     else:
    #         data[c] = np.zeros(len(data))
    # data = data.get_values()
    data, classlabels = loadData(rootPath + '/Norm/3PMNorm.csv')
    onetest = data[0] #异常
    twotest = data[1] #正常
    testD = [onetest, twotest]
    from numpy.random import choice
    for samplesize in [0.25, 0.5, 0.75]:
        """if的实验"""
        # f = open(dpath + "createStd5_depth_IF3{}.txt".format(samplesize), "w+")
        # f.write("ntrees\tavgDepth\tavgDepth2\n")
        # cn1 = 2. * (np.log(samplesize * 1000) + 0.5772156649) - 2 * (samplesize * 1000 - 1) / (samplesize * 1000)
        # print(cn1)
        # for i in range(2, 100):
        #     clf = IsolationForest(i, max_samples=samplesize)
        #     for j in range(50):
        #         clf.fit(data)
        #         a = clf.decision_function(testD)
        #         b = -np.log2(0.5 - a) * cn1
        #         f.write(str(i) + "\t" + str(b[0]) + "\t" + str(b[1]) + "\n")
        # f.close()
        # print('stat')
        f2 = open(rootPath + "/convergence/pima_depth_OIF_{}.txt".format(samplesize), "w+")
        f2.write("ntrees\tavgDepth\tavgDepth2\n")
        """原始版本，每个规模做50次"""
        # for i in range(50):
        #     forest = sT.statForest(data, 1200)
        #     estimaters = forest.fit(train_size=samplesize)
        #     for j in range(2, 100):
        #         if j <= 60:
        #             ee = estimaters[j:j + j]
        #         else:
        #             ee = estimaters[j:] + estimaters[:2 * j - 120]
        #         b = forest.predict(ee, testD)
        #         f2.write(str(j) + "\t" + str(b[0]) + "\t" + str(b[1]) + "\n")
        # f2.close()
        """新版本，每个规模做1次"""
        forest = sT.statForest(data, 6000)
        estimaters = forest.fit(train_size=samplesize)
        for j in range(2, 1000):
            ee = choice(estimaters,j,replace=False)
            b = forest.predict(ee, testD)
            f2.write(str(j) + "\t" + str(b[0]) + "\t" + str(b[1]) + "\n")
        f2.close()

def moons_circle_etc():
    from sklearn.datasets import make_moons, make_circles, make_classification, make_gaussian_quantiles
    np.random.seed(30)
    a, b = make_classification(500, 2, 2, 0, weights=[0.9, 0.1], random_state=2)
    rng = np.random.RandomState(2)
    a += 2 * rng.uniform(size=a.shape)
    linearly_separable = (a, b)

    data = [make_gaussian_quantiles(n_samples=900, n_classes=2, random_state=2),
            make_moons(900, noise=0.2, random_state=2),
            make_circles(900, noise=0.1, factor=0.2, random_state=2),
            linearly_separable]
    j = 1
    k = 0
    figure = 1


    newData = []
    for dd, ll in data:
        cc = []
        lll = []
        if j < 4:
            j+=1
            normal = [[], []]
            abnormal = [[], []]
            for i in range(900):
                if ll[i] == 0:
                    cc.append(dd[i])
                    normal[0].append(dd[i][0])
                    normal[1].append(dd[i][1])
                    lll.append(0)
                if ll[i] == 1:
                    if k == 2:
                        cc.append(dd[i])
                        lll.append(1)
                        abnormal[0].append(dd[i][0])
                        abnormal[1].append(dd[i][1])
                    k = (k + 1) % 9
            plt.subplot(2, 2, figure)
            plt.plot(normal[0], normal[1], '.', markersize=10, label='Normal')
            plt.plot(abnormal[0], abnormal[1], '+', markersize=10, label='Anomalous')
            plt.legend()
            figure += 1
        else:
            normal = [[], []]
            abnormal = [[], []]
            cc = list(dd)
            lll = list(ll)
            for i in range(len(cc)):
                if lll[i] == 0:
                    normal[0].append(cc[i][0])
                    normal[1].append(cc[i][1])
                else:
                    abnormal[0].append(cc[i][0])
                    abnormal[1].append(cc[i][1])
            plt.subplot(2, 2, figure)
            plt.plot(normal[0], normal[1], '.', markersize=10, label='Normal')
            plt.plot(abnormal[0], abnormal[1], '+', markersize=10, label='Anomalous')


        newData.append((cc,lll))
    plt.legend()
    plt.show()
    return newData

if __name__ == '__main__':
    # twoPointDepth2() #这里面有
    # twoPointDepth()

    files = pw.pathWalk('/media/xinxin/文档/UnionPayProject/randomForest/有效数据集/')  #对于这些数据，要记得改loadData的header
    # # files = pw.pathWalk('/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/elkiData/literature_ELKI_csv/')
    # # files = ['/home/xinxin/下载/vowels.mat']
    # # # funcdict = {"rTree":tree.DecisionTreeClassifier(max_features="log2"),"iForest":IsolationForest()}
    # # # 2IONorm.csv 1BCNorm.csv  6CDNorm.csv (credit)
    #
    # # Ds = moons_circle_etc()
    # # nameDs = 'gaussian_quantiles,make_moons, make_circles, make_classification'.split(',')
    # # print('LOF\tiForest\tIOT\tIOF')
    # print('IOF-auc\tIOF-time\tgradIOF-auc\tgradIOF-time')
    print('LOF\tiForest\tSciForest\tOIT\tOIF')

    # #这一段是用来找数据的代码
    # # path = '/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/AllData/norm_UCI/Norm/'
    # # # path = '/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/elkiData/literature_ELKI_csv/'
    # # files = ['1BCNorm.csv']
    #
    # # files = ['KDDCup99_1ofn.csv','KDDCup99_withoutdupl_1ofn.csv'] #,'PenDigits_withoutdupl_norm_v01.csv','WPBC_withoutdupl_norm.csv','Waveform_withoutdupl_norm_v07.csv'
    for fname in files[:1]:         #[1,5]:   #range(4):  #nameDs:    #     #
    #     # data, classlabels = Ds[fname]
    #     # print(nameDs[fname])
    #
    #     # import scipy.io as sio
    #     # matFile = '/home/xinxin/下载/vowels.mat'
    #     # vowMat = sio.loadmat(matFile)
    #     #
    #     # data = vowMat['X']
    #     # classlabels = vowMat['y'].ravel()
    #
    #     # if 'Shuttle' not in fname and flag:
    #     #     continue
    #
    #     # data, classlabels = loadData(fname)

        fname = '/media/xinxin/文档/UnionPayProject/randomForest/有效数据集/breast.txt'

        data, classlabels = loadData(fname)
        dname = fname.split("/")[-1].split('.')[0]
        print(dname)
    # #     # data, classlabels = createData(3,fname)

        # clf = LocalOutlierFactor(n_neighbors=10)
        # r = avgValidate(data, classlabels, clf,flag=2,algo='lof',dname=dname)
        # print(metrics.roc_auc_score(r[0], r[1]), end='\t')
        #
        # clf = IsolationForest()  #
        # r = avgValidate(data,classlabels,clf,flag=2,dname=dname)
        # print(metrics.roc_auc_score(r[0], r[1]),end='\t')
        #
        # r = statForestValidate(data, classlabels,nTrees=100,mode=0,dname=dname) #用SCiForest,SCiForest挑超平面系数时，绝对是用的-1,1
        # pred = 1. - np.array(r[1])
        # roc_auc = metrics.roc_auc_score(r[0], pred)
        # print(roc_auc, end='\t')

        r = avgStatValidate(data, classlabels,flag=2,dname=dname)  #用OIT算法，flag指导是否用十折交叉,1是10次运算
    #     pred = 1. - np.array(r[1])
    #     roc_auc = metrics.roc_auc_score(r[0], pred)
    # #     print(roc_auc)
    #     print(roc_auc, end='\t')
    # #
        r = statForestValidate(data,classlabels, 40,ifgrad=0,dname=dname) #用OIF，默认十折交叉
        # pred = 1. - np.array(r[1])
        # roc_auc = metrics.roc_auc_score(r[0], pred)
        # print(roc_auc)  # messure([r])

        # t1 = clock()
        # r = statForestValidate(data, classlabels, 40, ifgrad=1)
        # pred = 1. - np.array(r[1])
        # roc_auc = metrics.roc_auc_score(r[0], pred)
        # print(roc_auc, '\t', clock() - t1, end='\n')  # messure([r]) # 'statForest',

        # t1 = clock()
        # r = statForestValidate(data, classlabels, 100,mode=0)  #'statForest',
        # pred = 1. - np.array(r[1])
        # roc_auc = metrics.roc_auc_score(r[0], pred)
        # print(roc_auc, '\t', clock() - t1, end='\n')




    # if 1:
    #     for fname in [1]: #[:1]
    #         # data,classlabels = loadData(fname)
    #         # 下面部分专门测试正态分布在算法中的效果
    #         std1 = 3
    #         std2 = 5
    #         data, classlabels = createData(stdNormal=std1,stdOutlier=std2)
    #         r1 = avgValidate(data,classlabels,IsolationForest(contamination=0.2))
    #         s1 = messure(r1)
    #         r2 =avgStatValidate(data,classlabels)
    #         s2 = messure(r2)
    #
    #         #
    #         # y_pred = statForestValidate(data,5)
    #         clf = LocalOutlierFactor(n_neighbors=10,contamination=0.2)
    #         y_pred = clf.fit_predict(data)
    #         y_score = clf.negative_outlier_factor_
    #         a = clf._decision_function(data)
    #         print(a[:6])
    #         roc_auc = metrics.roc_auc_score(classlabels, a)
    #         print(roc_auc)
    #
    #         roc_auc = metrics.roc_auc_score(classlabels, y_score)
    #         print(roc_auc)
    #         s = messure([list(classlabels),list(y_pred)])
    #         # #
    #         print ("200:N(-3,%d),800:N(6,%d)" %(std2,std1)+"\tiforest\t"+s1)
    #         print ("200:N(-3,%d),800:N(6,%d)" %(std2,std1)+"\tstat\t"+s2)
    #         # print "200:N(-3,%d),800:N(6,%d)" %(std2,std1)+"\tstatforest\t"+s
    #         aa = 22#fname.split("\\")[-1]
    #
    #         # print aa+"\tiforest\t"+s1
    #         # print aa+"\tstat\t"+s2
    #         print(str(aa)+"\tlof\t"+s)
