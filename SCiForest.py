# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 10:40:43 2016

@author: Xin
"""

from math import log
import math
import numpy as np
import pandas as pd
from time import clock
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
import os.path as path
#from sklearn.preprocessing import MinMaxScaler

# rootPath = "/media/xinxin/文档/UnionPayProject/randomForest/data/StatTestData/AllData/norm_UCI/Norm/"
rootPath = '/media/xinxin/文档/UnionPayProject/randomForest/有效数据集/'


#下面直接返回归一化的数据了,以后的数据干脆都不要header好了
def loadData(fname):
    f = rootPath+fname
    data = None
    if path.exists(f+".txt"):
        data = pd.read_csv(f+".txt",header=None,na_values = '?')
    else:
        data = pd.read_csv(f+".csv",na_values = "?",header=None)
    classlabels = data[data.columns[-1]]
    del data[data.columns[-1]]
    data = data.bfill().ffill()
    for c in data.columns:
        maxd = max(data[c])
        mind = min(data[c])
        z = maxd - mind
        if z > 0:
            data[c] = (data[c] - mind)*1.0/ z
        else:
            data[c] = np.zeros(len(data))
    return data.get_values(), classlabels


def momenm(featV, bfact=0.1):
    featV = np.array(featV)
    SumX = np.sum(featV)
    SumX2 = np.sum(featV * featV)
    lenf = len(featV)
    step0 = int(np.ceil(0.01 * lenf))
    D = np.var(featV)
    A = 1.0*lenf
    Ec = featV[0]
    Enc = (SumX - featV[0]) / (lenf - 1.)
    Dnc = (SumX2 - featV[0] * featV[0]) / (lenf - 1.0) - Enc * Enc
    fraction = np.sqrt(A * (Dnc + A / (lenf - 1.0)))  #A/(i+1.0)  A/ (lenf-i-1.0)
    bestStat = (Enc - Ec) / fraction
    endF = step0
    step = step0
    bestSplit = (featV[0]+featV[1])/2.
    sumL = 0.0
    sumL2 = 0.0
    z = np.ceil(np.log10(lenf))
    while endF < lenf:
        for j in range(endF - step, endF):
            sumL += featV[j]
            sumL2 += featV[j] * featV[j]
        Ec = sumL / endF
        Enc = (SumX - sumL) / (lenf - endF)
        Dc = sumL2 / endF - Ec * Ec
        Dnc = (SumX2 - sumL2) / (lenf - endF) - Enc * Enc

        fraction = np.sqrt((Dc + A / endF) * (Dnc + A / (lenf - endF)))
        currStat = (Enc - Ec) / fraction
        gap = D if featV[endF] - bestSplit == 0 else (featV[endF] - bestSplit)
        grad = (currStat- bestStat) / gap
        v0 = grad*z
        # v0 =  10 if grad<0 else 0.1#*2#0.7*v0+
        # 用指数做步长明显不合适啊，一旦呈现下降趋势，那么可能就一次性跳过所有的数据了
        # step = int(np.ceil(2 **grade * step0))  #0.01*v0 +0.2*v0 +  0.5*v0 +  0.5*v0 +
        # step = int(np.ceil(5.*step0/(1.+ np.exp(v0))))
        step =  int(np.ceil(step0*3./(1.+np.exp(v0)))) if v0<0 else int(np.ceil(step0*(0.7-1.3/(1.+np.exp(v0)))))
        # step = int(np.ceil(v0*step0)) if grad<0 else int(np.ceil(v0*step0))#这种步长可能根本计算不出来，因为v0很小，所以得调参数
        if bestStat <= currStat:
            bestStat = currStat
            bestSplit = (featV[endF] + featV[endF - 1]) / 2.#
        endF += int(step)
    return bestStat*D, bestSplit

def calcStat(featValue,bfact=0.1):
    featV = np.array(featValue)
    currSplit = (featV[1]+featV[0])/2.0
    bestStat = -10.0
    SumX = np.sum(featV)
    SumX2 = np.sum(featV*featV)
    SumA = 0.0
    SumA2 = 0.0
    lenf = len(featV)
    Dx = np.clip((SumX2 - (SumX**2)/lenf)/lenf,0.0000001,100)
    Sdx = Dx**0.5
    #index = -1
    for i in range(lenf-1):
        SumA += featV[i]
        SumA2 += featV[i] * featV[i]
        Ec = SumA / (i+1.0)
        Enc = (SumX - SumA) / (lenf - (i+1.0))
        Dc = max(SumA2/(i+1.0) - Ec*Ec,0.0)
        Dnc = max((SumX2 - SumA2)/(lenf-i-1.0) - Enc*Enc,0.0)

        currStat = Dnc**0.5+Dc**0.5
        if currStat < bestStat:
            bestStat = currStat
            currSplit = (featV[i+1]+featV[i])/2.0
    return 1-bestStat*0.5/Sdx, currSplit  #*Dx

class final_stat:
    def __init__(self,bf=1.0,kfeat=-1):  #, label
        #self.label = label
        self.bf=bf
        self.kfeat = kfeat

    def fit(self,data):
        if self.kfeat == -1:
            self.kfeat = 10
        self.logLimit = log(len(data),2)#+0.5
        return self.build(data,0)

    def build(self,dataSet,depth):
        itemNum = len(dataSet)
        if itemNum == 2:
            return 1
        if itemNum <= 1:
            return 0

        bestStat = -1
        bestfeat = []
        bestSplit = -2
        bestCoef = []

        s = self.kfeat
        windowsize = len(dataSet[0])
        featWindow = [i for i in range(windowsize)]

        while((s > 0 or bestStat<0) and windowsize>1):
            chooseIndex = np.random.choice(featWindow, 2)  #此处2是一个参数，可以设置为别的
            featValue = np.zeros(itemNum)
            curCoef = []

            for index in chooseIndex:    #对于选出来的特征加权加和
                feat1 = [t[index] for t in dataSet]
                if len(set(feat1)) == 1:
                    if index in featWindow:
                        featWindow.remove(index)
                        windowsize -= 1
                    curCoef.append(0)
                else:
                    # curCoef.append((random.random()*2-1) / np.std(feat1))
                    curCoef.append(random.choice([1,-1]) / np.std(feat1))
                    featValue = featValue + curCoef[-1]*np.array(feat1)

            if set(curCoef) == set([0]):
                continue
            s -= 1

            featValue = sorted(featValue)
            if featValue[0] == featValue[-1]:
                continue
            curStat, cursplit = calcStat(featValue,self.bf)
            if curStat > bestStat:
                bestfeat = chooseIndex
                bestSplit = cursplit
                bestStat = curStat
                bestCoef = curCoef

        if bestStat == -1:  #可能都没有进入bestsplit的判断，说明全部相等
            return (2.0*log(itemNum-1)+1.1544313298-2.0*(itemNum-1)/itemNum)#+0.1
        data1 = []
        data2=[]

        selectFeat = np.array([[t[i] for i in bestfeat] for t in dataSet])
        feat = np.sum(bestCoef*selectFeat,axis=1)

        if min(feat) == bestSplit:
            for i in range(itemNum):
                if feat[i] == bestSplit:
                    data1.append(dataSet[i])
                else: data2.append(dataSet[i])
            bestSplit = 1.0001*bestSplit  #该算法是严格的左边小于分割点
        else:
            for i in range(itemNum):
                if feat[i] < bestSplit:
                    data1.append(dataSet[i])
                else: data2.append(dataSet[i])

        subtree = [bestfeat,bestCoef,bestSplit,max(feat)-min(feat)]
        subtree.append(self.build(data1,depth+1))
        subtree.append(self.build(data2,depth+1))

        return subtree

    def classifyVec(self,inputTree, testVec):  #此处的featlabel不设置为self，这样子测试的特征顺序就可以和训练的不一样了
        featName = inputTree[0]
        coef = np.array(inputTree[1])
        depth = 0

        value = np.array([testVec[i] for i in featName])
        mixFeat = sum(coef*value)

        if mixFeat < inputTree[2]:
            depth += (0 if mixFeat <= (inputTree[2] - inputTree[3]) else 1)
            if type(inputTree[4]).__name__ == 'list':
                depth += self.classifyVec(inputTree[4], testVec)
            else:
                depth += inputTree[4]
        else:
            depth += (0 if mixFeat >= (inputTree[2] + inputTree[3]) else 1)
            if type(inputTree[5]).__name__ == 'list':
                depth += self.classifyVec(inputTree[5], testVec)
            else:
                depth += inputTree[5]
        return depth

    def classify(self,inputTree, testdata):
        predict = []

        for i in testdata:
            # 返回分类标签的版本
            # clabel = 0 if self.classifyVec(inputTree, i) > self.logLimit else 1

            # 返回路径长度的版本
            clabel = self.classifyVec(inputTree, i)
            predict.append(clabel)

        return predict


class statForest:
    def __init__(self, data, nTree):  #k树的棵树   , label   #, N
        self.data = data
        self.nTree = nTree
        self.sTree = final_stat()
        # self.lim = log(len(data),2)  #-1 0.3
        #有kfeat，bf可以设置
        #self.samplePropotion = N

    def fit(self,train_size = 0.3):
        estimaters = []
        N = min(len(self.data)-1, 256)  #
        for i in range(self.nTree):
            t, _ = train_test_split(self.data,train_size=N)  #self.samplePropotion
            estimaters.append(self.sTree.fit(t))
        return estimaters

    def classify(self,inputTree, testdata):
        predict = []
        for i in testdata:
            clabel = self.sTree.classifyVec(inputTree, i)
            predict.append(clabel)
        return predict

    def predict(self,estimaters,testData):
        resultlist = []
        for e in estimaters:
            resultlist.append(self.classify(e,testData))
        vote = np.average(np.array(resultlist),axis=0)

        # vote = map(lambda t: 0 if t > self.lim else 1,vote)
        return vote

if __name__ == '__main__':
    strings = '1BCNorm,breast_Norm,2IONorm,3PMNorm,8glassesNorm,breast_diagnostic_Norm'
    #'breast-cancer.csv,ionosphere.txt,pima_diabetes.txt,Arraythmia.csv,satellite1.csv,creditCardDefault.csv,climateSimulation.csv,CTG.csv,sonar.csv'  #,pimaKnn6.csv,Arraythmia3.csv
    for fname in strings.split(',')[-1:]:
        data,c = loadData(fname)
        # from sklearn.metrics import roc_auc_score

        print(len(data),len(data[0]))

        forest = statForest(data,5)
        estimaters = forest.fit()

        print(estimaters[0])

        # results = forest.predict(estimaters,data)
        #
        # c = 1-np.array(c)
        # print(roc_auc_score(c, results))

        # print results[:10], type(results)


