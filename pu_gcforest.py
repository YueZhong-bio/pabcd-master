#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
pu_forest.py
Desc :  PU-Learning GCforest algorithm used to predict circRNA-disease associations.
Usage: ./pu_gcforest.py directory-contains-feature-files outDir randomSeed
E.g. : ./pu_gcforest.py ./features ./gcforest-outdir 10
Coder: zhongyue, etc
Created date: 20181208
'''

import argparse
import numpy as np

import sys
from os import listdir
from os.path import isfile, join

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
sys.path.insert(0, "lib")

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json

if(len(sys.argv) != 4):
    sys.exit("Usage: %s directory-for-input-files outdir randomSeed\n" %(sys.argv[0]))

inpath = sys.argv[1]
outdir = sys.argv[2]
rs = int(sys.argv[3])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args

def calPRAUC(ranks, nTPs, topN):
    cumPRAUC = 0
    posRecalls = list()
    for i in range(topN):
        if(ranks[i] < nTPs):
            posRecalls.append(1)
        else:
            posRecalls.append(0)

    curSum = posRecalls[0]
    prevRecall = round(posRecalls[0] / nTPs, 4)
    prevPrec = round(posRecalls[0], 4)
    for i in range(1, topN):
        curSum += posRecalls[i]
        recall = round(curSum / nTPs, 4)
        prec   = round(curSum / (i+1), 4)
        cumPRAUC += ((recall - prevRecall) * (prevPrec + prec) / 2)
        prevRecall = recall
        prevPrec = prec

    cumPRAUC = round(cumPRAUC, 4)
    return cumPRAUC

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 1
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 100, "max_depth": 100,
    #          "objective": "multi:softprob","num_class":2, "silent": True, "nthread": -1, "learning_rate": 0.1,"class_weight": {0:1,1:64}})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression", "class_weight": {0:1,1:32}})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    # args = parse_args()
    # if args.model is None:
    # config = get_toy_config()
    # else:
    #     config = load_json(args.model)

    # gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.
    dFeatures = [f for f in listdir(inpath) if isfile(join(inpath, f))]
    for df in dFeatures:
        print("Processing %s" % (df))
        dId = df.split('_')[0]
        pf = "/".join((inpath, df))  # processing input file: df

        outfile = ".".join((dId, "txt"))
        of = "/".join((outdir, outfile))
        d = np.loadtxt(pf, delimiter=',')
        p = d[d[:, 24] == 1, :]
        u = d[d[:, 24] == 0, :]
        x_p = p[:, 0:24]
        y_p = p[:, 24]
        x_u = u[:, 0:24]
        X_n = x_u[0]
        y_u = u[:, 24]
        x_u_train, x_u_test, y_u_train, y_u_test = train_test_split(x_u, y_u, test_size=0.2, random_state=1)
        #训练分类器选取可靠负样本
        i = 0
        nfolds = 5
        for i in range(nfolds):
            x_p_s, x_p_test, y_p_s, y_p_test = train_test_split(x_p, y_p, test_size=0.15, random_state=1)
            x_u_s = np.concatenate((x_p_test, x_u), axis=0)
            y_u_s = np.concatenate((y_p_test, y_u), axis=0)
            y_u_s = np.zeros(y_u_s.shape[0])

            x = np.concatenate((x_p_s, x_u_s), axis=0)
            y = np.concatenate((y_p_s, y_u_s), axis=0)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
            # scaler = StandardScaler().fit(X_train)
            # X_train_transformed = scaler.transform(X_train)
            # X_test_transformed = scaler.transform(X_test)
            config = get_toy_config()
            gc = GCForest(config)
            gc.fit_transform(x_train, y_train)

            scores = gc.predict_proba(x_u_test)[:,0]
            orderScores = np.argsort(-scores)
            orderList = [str(item) for item in orderScores]
            orderStr = ','.join(orderList)
            top = int(y_u_test.shape[0]*0.25)
            topNIndex = orderScores[:top]
            t = 0
            while t < top:
                index = topNIndex[t]
                x_n = x_u[index]
                X_n = np.vstack((X_n,x_n))
                t += 1
        X_n = X_n[1:,:]
        X_n = np.unique(X_n, axis=0)
        Y_n = np.zeros(X_n.shape[0])
        X = np.concatenate((x_p, X_n), axis=0)
        Y = np.concatenate((y_p, Y_n), axis=0)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #利用正样本和可靠负样本重新训练分类器
        config = get_toy_config()
        gc = GCForest(config)
        gc.fit_transform(x_train, y_train)
        y_pred = gc.predict(x_test)
        # acc = accuracy_score(y_test, y_pred)
        # print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

        i = 0
        nfolds = 5
        eRecalls = np.zeros(nfolds)
        ePrecisions = np.zeros(nfolds)
        ePRAUCs = np.zeros(nfolds)
        for i in range(nfolds):
            x_p_train, x_p_test, y_p_train, y_p_test = train_test_split(x_p, y_p, test_size=0.2)
            x_u_train, x_u_test, y_u_train, y_u_test = train_test_split(x_u, y_u, test_size=0.2)
            X_test = np.concatenate((x_p_test, x_u_test), axis=0)
            Y_test = np.concatenate((y_p_test, y_u_test), axis=0)

            scores = gc.predict_proba(X_test)[:, 1]
            orderScores = np.argsort(-scores)
            orderList = [str(item) for item in orderScores]
            orderStr = ','.join(orderList)
            truePosIndex = np.array(range(y_p_test.shape[0]))  ###正样本的数量构建数组

            # xz = list(range(0,500))
            # yz = np.array(np.zeros(500))
            # for t in range(0, 500):
            topNIndex = orderScores[:500]  ###预测结果前500个
            truePosRecall = np.intersect1d(topNIndex, truePosIndex, assume_unique=True)  ###预测结果和正样本的交集
                # recall = truePosRecall.shape[0] / truePosIndex.shape[0]
                # yz[t] = recall
            # plt.figure(1)
            # plt.plot(xz,yz,label="gcforest",color="red",linewidth=2)
            # plt.show()

            #print(truePosRecall.shape[0],truePosIndex.shape[0])
            recall = truePosRecall.shape[0] / truePosIndex.shape[0]
            precision = truePosRecall.shape[0] / 500
            prauc = calPRAUC(orderScores, y_p.shape[0], 500)
            #print(recall, precision, prauc)
            eRecalls[i] = recall
            ePrecisions[i] = precision
            ePRAUCs[i] = prauc
        mRecall = np.mean(eRecalls)
        stdRecall = np.std(eRecalls)
        mPrec = np.mean(ePrecisions)
        stdPrec = np.std(ePrecisions)
        mPRAUC = np.mean(ePRAUCs)
        stdPRAUC = np.std(ePRAUCs)
        recallList = [str(item) for item in eRecalls]
        precList = [str(item) for item in ePrecisions]
        praucList = [str(item) for item in ePRAUCs]
        recallStr = ','.join(recallList)
        precStr = ','.join(precList)
        praucStr = ','.join(praucList)
        with open(of, "a") as output:
            output.write("%s-RandomState%d, mean+-std recall:%.4f,%.4f\n" % (dId, rs, mRecall, stdRecall))
            output.write("%s-RandomState%d, mean+-std precision:%.4f,%.4f\n" % (dId, rs, mPrec, stdPrec))
            output.write("%s-RandomState%d, mean+-std prauc:%.4f,%.4f\n" % (dId, rs, mPRAUC, stdPRAUC))
            output.write("%s-RandomState%d, 5-fold cv recall:%s\n" % (dId, rs, recallStr))
            output.write("%s-RandomState%d, 5-fold cv precision:%s\n" % (dId, rs, precStr))
            output.write("%s-RandomState%d, 5-fold cv prauc:%s\n" % (dId, rs, praucStr))
            output.write("END\n")
