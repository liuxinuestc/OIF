# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:47:46 2016

@author: Administrator
"""

import matplotlib.pyplot as plt
from numpy import array

# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")

decisionNode = dict(boxstyle='circle', fc='0.9') #boxstyle='circle',
leafNode = dict(boxstyle='round', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode( centerPt, parentPt, nodeType,nodeTxt='.'):  # 节点名称，节点位置，指向节点的节点位置，节点类型，画出节点与指向节点的线
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, \
                            textcoords='axes fraction', bbox=nodeType, va='center', ha='center',  arrowprops=arrow_args)  #


def getNumLeafs(myTree):  # 获得树的叶子数，即宽度
    numLeafs = 0
    first = myTree
    for key in first[4:]:
        if type(key).__name__ == 'list':
            numLeafs += getNumLeafs(key)
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):  # 获得树的深度
    maxDepth = 0
    second = myTree
    for key in second[4:]:
        if type(key).__name__ == 'list':
            thisDepth = 1 + getTreeDepth(key)
        else:
            thisDepth = 1
        if thisDepth > maxDepth:  maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):  # 在线上加注释，从而x，y轴坐标是线段两点的中点
    xMid = (parentPt[0] + cntrPt[0]) / 2.0
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt=' '):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)

    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(cntrPt, parentPt, decisionNode)
    second = myTree
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in second[4:]:
        if type(key).__name__ == 'list':
            plotTree(key, cntrPt)
            # plotTree(key, cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode((plotTree.xOff, plotTree.yOff), cntrPt, leafNode,int(key*1000)/1000.0)
            # plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white',figsize=(24,24))
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    print(plotTree.totalW)
    plotTree(inTree, (0.5, 1.0), ' ')
    plt.show()


if __name__ == '__main__':
    tree = []

    createPlot(tree)