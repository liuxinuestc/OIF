# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:47:46 2016

@author: Administrator
"""

import matplotlib.pyplot as plt

# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")

decisionNode = dict(boxstyle='circle', fc='0.8')
leafNode = dict(boxstyle='round', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode( centerPt, parentPt, nodeType,nodeTxt='o'):  # 节点名称，节点位置，指向节点的节点位置，节点类型，画出节点与指向节点的线
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, \
                            textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):  # 获得树的叶子数，即宽度
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    second = myTree[firstStr]
    for key in second[1:]:
        if type(key).__name__ == 'dict':
            numLeafs += getNumLeafs(key)
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):  # 获得树的深度
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    second = myTree[firstStr]
    for key in second:
        if type(key).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(key)
        else:
            thisDepth = 1
        if thisDepth > maxDepth:  maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):  # 树数据
    listOfTrees = [{'width': {2: 'a', 3: {'length': {1: 'b', 3: 'c', 4: 'c'}}, 4: 'c', 5: 'b'}}, \
                   {'width': {0: {'col': {0: 'a', 1: {'length': {0: 'b', 1: 'c', 2: 'a'}}, 2: 'b', 3: 'c'}},
                              1: {'length': {0: 'a', 2: 'b', 1: 'c'}}, 2: 'c'}}]
    # {0:{0:1,1:{2:{1:1,4:3},4:{3:2,5:{4:0,1:5}}}}}
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):  # 在线上加注释，从而x，y轴坐标是线段两点的中点
    xMid = (parentPt[0] + cntrPt[0]) / 2.0
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt=' '):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]

    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(cntrPt, parentPt, decisionNode)
    second = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in second[1:]:
        if type(key).__name__ == 'dict':
            plotTree(key, cntrPt)
            # plotTree(key, cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode((plotTree.xOff, plotTree.yOff), cntrPt, leafNode,int(key*1000)/1000.0)
            # plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
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
    tree = {0: [0.43135027687100003, {18: [0.2279929082005, {21: [0.378198294243, {19: [0.11933584843050001, {26: [0.1729233226835, {24: [0.30958198507550005, {27: [0.172611683849, {9: [0.14469250210599999, {25: [0.061700187249549995, 10.207392357586556, 9]}, {0: [0.28037294713450001, 10.207392357586556, 9]}]}, {20: [0.228744219139, {14: [0.149947309379, 10.207392357586556, 9]}, {4: [0.25918570009949998, 10.207392357586556, 10]}]}]}, {29: [0.125213170668, {18: [0.12513367479045001, 9, {12: [0.070654478631650003, 10.207392357586556, 9]}]}, {0: [0.20228122485700001, {2: [0.1434938843205, 10.207392357586556, 9]}, {27: [0.317113402062, 11.327020052039781, 10.207392357586556]}]}]}]}, {7: [0.25534294234599997, {15: [0.1755790548865, {9: [0.159540859309, {21: [0.25013326225999999, 10, 9]}, {6: [0.15149953139649999, 9, 10]}]}, {20: [0.26342938456050002, 8, 9]}]}, 7]}]}, {24: [0.55689097272650001, {14: [0.2687221674545, {18: [0.13955648111649999, 8, {12: [0.040828346605099999, 8, 9]}]}, 7]}, 6]}]}, {5: [0.26492239739900003, {24: [0.36241167536149999, {2: [0.23077188860500003, {19: [0.043710183381049997, 7, 8]}, {18: [0.10726346597620001, {1: [0.39228948258399998, {19: [0.030616475270549996, 9, 10]}, 9]}, {10: [0.063860220894450009, 9, 9]}]}]}, {22: [0.2512575327455, {4: [0.43337546267049998, {11: [0.34693246110300002, 9, 8]}, {2: [0.16481238338750001, 8, 9]}]}, {6: [0.14854732896, 8, 7]}]}]}, {26: [0.74369009584650003, {9: [0.37131423757350002, {25: [0.33623424629599996, {16: [0.1249873737373, 9, 8]}, {24: [0.65363534306299997, {27: [0.45567010309300005, 9, 10.207392357586556]}, 8]}]}, {11: [0.27808963932100006, {13: [0.20888759390199999, {8: [0.48863636363649998, 10, 10.207392357586556]}, 8]}, 8]}]}, 5]}]}]}, {26: [0.19069488817899999, {25: [0.10095953274949999, {21: [0.251066098081, {10: [0.1224153539741, {4: [0.23296018777650002, 7, 8]}, {15: [0.14610057980594998, 8, 7]}]}, {19: [0.12114962066255, {2: [0.23218851496100001, 8, 8]}, 6]}]}, {13: [0.039751736091649996, {24: [0.33916661163549999, 6, {27: [0.23841924398600001, 8, 8]}]}, {15: [0.39995343527499999, {10: [0.1291870360315, 8, 8]}, 6]}]}]}, {10: [0.093699076588849989, {25: [0.34486907083500007, {13: [0.034932891045550002, {11: [0.14752033239029999, 7, {4: [0.43590322289449995, 8, 9]}]}, 6]}, 6]}, {20: [0.21807186054799998, 6, 6]}]}]}]}, {23: [0.50145005898549999, {18: [0.50610682726399991, {2: [0.5698293137999999, {18: [0.19759948218600001, {7: [0.34731610337999996, {24: [0.25279006801799997, {8: [0.23888888888904997, 8, 9]}, {15: [0.2376152852465, {23: [0.23036276051899998, 10.207392357586556, 10.207392357586556]}, 8]}]}, {6: [0.3656279287725, {9: [0.19155433866900001, {10: [0.1833242802825, 10.207392357586556, 9]}, 9]}, 8]}]}, {2: [0.5017621449795, {11: [0.20247303748249998, 8, 7]}, {12: [0.20534325967149997, 8, 7]}]}]}, {7: [0.58846918489049993, {7: [0.44423459244550001, 6, {17: [0.49696912293999995, {5: [0.39773633519399998, 9, 8]}, 7]}]}, {25: [0.22810489856499999, 6, {14: [0.16692728694300002, 8, 8]}]}]}]}, {18: [0.81975009849699998, 5, 4]}]}, {18: [0.34499352732599997, {10: [0.63932645301449997, {22: [0.71388017331499998, {1: [0.61024687182949999, 7, 6]}, {4: [0.43662544010100002, 7, 6]}]}, 4]}, 3]}]}]}
    createPlot(tree)