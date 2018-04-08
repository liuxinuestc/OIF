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


    """
    fsave = open(rootPath+"minMaxedData\\"+fname+"_nor.txt","w")
    for j in range(len(data)):
        string = "\t".join(map(str,data[j]))+"\n"
        fsave.write(string)
    fsave.close()"""

def momenm(featV, count=0):
    featV = np.array(featV)
    SumX = np.sum(featV)
    SumX2 = np.sum(featV * featV)
    lenf = len(featV)
    step0 = int(np.ceil(0.01 * lenf))
    D = np.var(featV)
    Ec = featV[0]
    Enc = (SumX - featV[0]) / (lenf - 1.)
    Dnc = (SumX2 - featV[0] * featV[0]) / (lenf - 1.0) - Enc * Enc
    Dnc = max(Dnc,0.0000001)
    fraction = Dnc * (lenf - 1.0)  # A/(i+1.0)  A/ (lenf-i-1.0)  Dc*endF+Dnc*(lenf-endF)

    bestStat = np.exp(Enc - Ec) / fraction

    endF = step0
    step = step0
    bestSplit = (featV[0] + featV[1]) / 2.
    sumL = 0.0

    z = np.ceil(np.log10(lenf))

    while endF < lenf:
        for j in range(endF - step, endF):
            sumL += featV[j]

        Ec = sumL / endF
        Enc = (SumX - sumL) / (lenf - endF)

        fraction = SumX2 - Ec * Ec * endF - Enc * Enc * (lenf - endF)

        fraction = max(0.000001,fraction)
        currStat = np.exp(Enc-Ec) / fraction

        gap = D if featV[endF] - bestSplit < 0.000001 else (featV[endF] - bestSplit)

        grad = (currStat - bestStat) / gap
        v0 = grad*z
        v0 = np.clip(v0,-10,10)

        # step = int(np.ceil(2 **grade * step0))  #0.01*v0 +0.2*v0 +  0.5*v0 +  0.5*v0 +
        # step = int(np.ceil(5.*step0/(1.+ np.exp(v0))))
        step = int(np.ceil(step0*10./(1.+np.exp(v0)))) if v0 < 0 else int(np.ceil(step0*(0.8-1.5/(1.+np.exp(v0)))))
        step = max(2,step)
        # step = int(np.ceil(v0*step0)) if grad<0 else int(np.ceil(v0*step0))#这种步长可能根本计算不出来，因为v0很小，所以得调参数
        if bestStat <= currStat:
            bestStat = currStat
            bestSplit = (featV[endF] + featV[endF - 1]) / 2.#
        endF += int(step)
        count += 1
    # print(count1,end='\t')
    return bestStat*D*D, bestSplit, count

ffff = 1
def calcStat(featValue,count=0):
    global ffff
    if ffff !=0:
        ffff = 0
        # print('hello')
    featV = np.array(featValue)
    currSplit = (featV[1]+featV[0])/2.0
    bestStat = -10.0
    SumX = np.sum(featV)
    Dx = np.var(featV)

    SumX2 = np.sum(featV*featV)
    SumA = 0.0
    SumA2 = 0.0
    lenf = len(featV)
    count += (lenf-1)

    for i in range(lenf-1):
        SumA += featV[i]
        SumA2 += featV[i] * featV[i]
        Ec = SumA / (i+1.0)
        Enc = (SumX - SumA) / (lenf - (i+1.0))

        Dc = SumA2/(i+1.0) - Ec*Ec
        Dc = Dc if Dc >0.0 else 0.0000001
        Dnc = (SumX2 - SumA2)/(lenf-i-1.0) - Enc*Enc
        Dnc = Dnc if Dnc >0.0 else 0.0000001
        # Dc = SumA2 - Ec*Ec*(i+1.0)
        # Dnc = (SumX2 - SumA2) - Enc*Enc*(lenf-i-1.0)

        # fraction = np.sqrt((Dc + A*Dx/(i+1.0))*(Dnc + A*Dx/(lenf-i-1.0)))  #A/(i+1.0)  A/ (lenf-i-1.0)
        # fraction = np.sqrt((Dc + (lenf - i - 1.0)/A) * (Dnc + (i + 1.0)/A))
        # fraction = Dc**((i+1.0)/A) + Dnc**((lenf-i-1.0)/A)
        # fraction = np.sqrt((A / (i + 1.0)) * (A / (lenf - i - 1.0))) # A/(i+1.0)  A/ (lenf-i-1.0)
        # fraction = np.sqrt((Dc + 1.0)*(Dnc + 1.0))
        fraction = Dc*(i+1.0)+Dnc*(lenf-i-1) #等价于SumX2 - Ec * SumA - Enc * (SumX-SumA)
        # fraction = (Dc+Dnc)/2
        # currStat = (Enc - Ec)/ fraction #if fraction != 0 else np.sqrt((Dc + 0.001)*(Dnc + 0.001)) #公式里用的方差

        # currStat =abs(Enc / np.exp(Dnc**0.5) - Ec / np.exp(Dc**0.5))/ fraction
        currStat = np.exp(Enc-Ec) / fraction #*np.exp(Enc-Ec)
        # currStat = (Enc - Ec) * ((i + 1.0) * (lenf - i - 1.0)) ** 0.5 / fraction
        #currStat = (Enc - Ec)/ np.sqrt((Dc + 0.001)*(Dnc  + 0.001))#fraction if fraction != 0 else -1  * abs(lenf-2-2*i)
        #currStat = (Enc - Ec) * StdX / np.sqrt((Dc**0.5+bfact/(i+1))*(Dnc**0.5 + bfact / (len-i-1))) #公式里用的标准差
        #currStat = (Enc - Ec) * Dx / np.sqrt((Dc+bfact/(i+1))*(Dnc + bfact / (lenf-i-1))) #原始公式不得改动

        if currStat > bestStat:
            bestStat = currStat
            currSplit = (featV[i+1]+featV[i])/2.0
    # print(count,end='\t')

    #公式三与公式四
    return bestStat*Dx*Dx*lenf, currSplit, count #Dx*

func =[calcStat,momenm]

class final_stat:
    def __init__(self,mode=0,bf=1.0,kfeat=-1):  #, label
        #self.label = label
        self.mode = mode
        self.bf=bf
        self.kfeat = kfeat
        self.count = 0

    def fit(self,data):
        if self.kfeat == -1:
            self.kfeat = np.ceil(log(len(data[0]),2))
        self.logLimit = np.ceil(log(len(data),2))
        return self.build(data,0)

    def build(self,dataSet,depth):
        itemNum = len(dataSet)
        if itemNum == 2:
            return depth+1
        if itemNum == 1:
            return depth
        if depth > self.logLimit:
            return depth + 2.0*log(itemNum-1) + 1.1544313298 - 2.0*(itemNum-1)/itemNum
        bestStat = -1
        bestfeat = -1
        bestSplit = -2
        s = self.kfeat
        windowsize = len(dataSet[0])
        featWindow = [i for i in range(windowsize)]

        while(s > 0 or (bestStat<0 and windowsize>0)):
            chooseIndex = random.choice(featWindow)
            featValue = [t[chooseIndex] for t in dataSet]

            featWindow.remove(chooseIndex)
            s -= 1;windowsize -= 1

            featValue = sorted(featValue)   #,所有抽样不抽样都统一了，都是输入全部值，抽样的都在小函数内部解决
            if featValue[0] == featValue[-1]:
                continue
            # if itemNum >=100:
            #     curStat, cursplit, self.count = func[self.mode](featValue,self.count)
            # else:
            #     curStat, cursplit, self.count = func[0](featValue, self.count)
            curStat, cursplit, self.count = func[self.mode](featValue, self.count)
            # curStat, cursplit = momenm(featValue)
            if curStat > bestStat:
                bestfeat = chooseIndex
                bestSplit = cursplit
                bestStat = curStat

        if bestStat == -1:  #可能都没有进入bestsplit的判断，说明全部相等
            return depth + 2.0*log(itemNum-1) + 1.1544313298 - 2.0*(itemNum-1)/itemNum+0.1
        data1 = []
        data2=[]

        feat = [i[bestfeat] for i in dataSet]
        if min(feat) == bestSplit:
            for i in dataSet:
                if i[bestfeat] == bestSplit:
                    data1.append(i)
                else: data2.append(i)
            bestSplit = bestSplit+0.001*bestSplit
        else:
            for i in dataSet:
                if i[bestfeat] < bestSplit:
                    data1.append(i)
                else: data2.append(i)

        subtree = [bestSplit]
        subtree.append(self.build(data1,depth+1))   #mixedBuild是固定的calcstat
        subtree.append(self.build(data2,depth+1))
        myTree = {bestfeat: subtree}
        return myTree

    def classifyVec(self,inputTree, testVec):  #此处的featlabel不设置为self，这样子测试的特征顺序就可以和训练的不一样了
        featName = list(inputTree.keys())[0]
        classLabel = None
        if testVec[featName] < inputTree[featName][0]:
            if type( inputTree[featName][1]).__name__ =='dict':
                classLabel = self.classifyVec(inputTree[featName][1], testVec)
            else:
                classLabel = inputTree[featName][1]
        else:
            if type( inputTree[featName][2]).__name__ =='dict':
                classLabel = self.classifyVec(inputTree[featName][2], testVec)
            else:
                classLabel = inputTree[featName][2]
        return classLabel  #返回的是叶子点的深度

    def classify(self,inputTree, testdata):
        predict = []
        for i in testdata:
            clabel = self.classifyVec(inputTree, i) # > self.logLimit
            # clabel = 0 if self.classifyVec(inputTree, i) > self.logLimit else 1
            predict.append(clabel)
        return predict   #返回的是叶子点的标签

class statForest:
    def __init__(self, data, nTree,mode=0):  #k树的棵树   , label   #, N
        self.data = data
        self.nTree = nTree
        self.sTree = final_stat(mode=mode)
        self.lim = log(len(data)*0.3,2)#0.9 -1 0.3  #该参数对于classify返回是叶子深度的方法没有作用
        #有kfeat，bf可以设置
        #self.samplePropotion = N

    def fit(self,train_size=0.3):
        estimaters = []
        N = min(200,int(train_size*len(self.data)))
        for i in range(self.nTree):
            t, _ = train_test_split(self.data,train_size=train_size)  #self.samplePropotion
            estimaters.append(self.sTree.fit(t))
        return estimaters

    def classify(self,inputTree, testdata):
        predict = []
        for i in testdata:
            clabel = self.sTree.classifyVec(inputTree, i) #得到数据在各个树上的深度
            predict.append(clabel)
        return predict

    def predict(self,estimaters,testData):
        resultlist = []
        for e in estimaters:
            resultlist.append(self.classify(e,testData))
        vote = np.mean(np.array(resultlist),axis=0)
        return list(vote)
        vote = list(map(lambda t: 0 if t > self.lim else 1,vote))
        return vote


if __name__ == '__main__':
    strings = 'breast_Norm,2IONorm,8glassesNorm,breast_diagnostic_Norm'
    #'breast-cancer.csv,ionosphere.txt,pima_diabetes.txt,Arraythmia.csv,satellite1.csv,creditCardDefault.csv,climateSimulation.csv,CTG.csv,sonar.csv'  #,pimaKnn6.csv,Arraythmia3.csv
    print('data\tlength\toriginal\tgrad')
    for fname in strings.split(',')[-1:]:
        data,c = loadData(fname)
        # data = pd.read_csv(rootPath+fname+'.csv',header=None)

        lend = len(data)

        print (fname,'\t',lend)
        # for i in data.columns[:-1]:
        #     calcStat(data[i])
        #     momenm(data[i])
        #     print()

        ft = statForest(data,5)
        ff = ft.fit()
        print(ff[0])

        # predict = ft.predict(ff,data)
        # x = 0

        # import sklearn.tree as tree
        # clf = tree.DecisionTreeClassifier()
        # clf.fit(data[:0.7*len(data)],c[:int(0.7*lend)])
        # predict = clf.predict(data[int(0.7*lend):])

        # testc = c[int(0.7*lend):]
        # x += len(testc[testc==predict])
        # print int(0.3*lend)

        # st = final_stat()
        # tree = st.fit(data)
        # predict = st.classify(tree,data)
        # # preTrans = map(lambda t:'g' if t==0 else 'b', predict)
        #
        # if c[0] == 'g' or c[0]=='b':
        #     c = np.array(map(lambda t: 0 if t=='g' else 1, c))

        # x += len(c[c==predict])
        # print("right",x)
        # print ("predict normal",sum(int(j==0) for j in predict))
        # z = sum(int(i==1 and j==1) for i,j in zip(c,predict))
        # print('right anomaly',z)
        # print ('anomaly',len(c[c==1]))





        # print len(data),len(data[0]),type(data)
        # forest = statForest(data,13)
        # estimaters = forest.fit()
        # results = forest.predict(estimaters,data)
        #
        # print results[:10], type(results)

