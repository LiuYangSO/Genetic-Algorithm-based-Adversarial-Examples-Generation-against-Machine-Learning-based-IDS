import os
from time import sleep
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import tree, svm, neighbors, metrics
from sklearn import tree
import graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from labels import data_headers, attack_dict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn import datasets,ensemble


# Setup global variables from config
# ATTACK = "teardrop"
# First Step CREATE　MODELS
#
class Model:

    def __init__(self, debug, attack, save_model):
        self.DEBUG = debug
        self.ATTACK = attack
        self.SAVE = save_model
        self.num = 0 # calculate the attack num of the specific type
        self.add = 0 # calculate the whole num of the sample in the dataset
    # Create a label for attack types that are being focused on
    # count the num of the attack samples
    def label_attack(self, row):
        if row['attack'] == self.ATTACK:
            self.add = self.add +1

            return 1
        else:
            return 0
    # --- DATA PREPARATION --- #
    def generate_model(self,num):
        # Open the dataset: set the train part and the test part
        test_df = pd.read_csv("Datasets/NSL-KDD/KDDTest+.txt", header=None, names=data_headers)
        # use the data_headers for each of the feature
        test_df = test_df.astype({"protocol_type": str, "service": str, "flag": str})
        # add colum: type_label : train
        test_df['type_label'] = "test"
        test_df = test_df.drop('type_label', axis=1)
        # pick the data that we want and then dropoff the label
        # Label attacks
        test_attack = test_df.apply(self.label_attack, axis=1)
        test_df["attack_label"] = test_attack

        # DROP PROTOCOL TYPE SERVICE FLAG
        test_df = test_df.drop('protocol_type', axis=1)
        test_df = test_df.drop('service', axis=1)
        test_df = test_df.drop('flag', axis=1)
        # this is the core part of an attack, so drop these could keep the dataset to be the attacks

        # Train part
        x_test = test_df.drop(["unknown", "attack", "attack_label"], axis=1)
        y_test = test_df['attack_label']
        train_Full = pd.read_csv("Datasets/NSL-KDD/KDDTrain+_20Percent.txt", header=None, names=data_headers)
        N = train_Full.size # mark the whole num of the train samples
        print('Datasets from: "Datasets/NSL-KDD/KDDTrain+.txt"')

        train_Full = train_Full.astype({"protocol_type": str, "service": str, "flag": str})
        attack = train_Full.apply(self.label_attack, axis=1)
        train_Full["attack_label"] = attack
        train_Full = train_Full.drop('protocol_type', axis=1)
        train_Full = train_Full.drop('service', axis=1)
        train_Full = train_Full.drop('flag', axis=1)

        # Here is the part that I use the 10 cross fold validation to train the final classifiers of each machine learning algorithms
        train_Full0 = train_Full.sample(n=N // 10, replace=True)
        # seperate the whole rain dataset into 10, and pick one as the train data for the classifier
        # repeat 10 times and test the accuracy of each model, select the best model for as the classifier
        self.y0 = train_Full0['attack_label']
        self.x0 = train_Full0.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full1 = train_Full.sample(n=N // 10, replace=True)
        self.y1 = train_Full1['attack_label']
        self.x1 = train_Full1.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full2 = train_Full.sample(n=N // 10, replace=True)
        self.y2 = train_Full2['attack_label']
        self.x2 = train_Full2.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full3 = train_Full.sample(n=N // 10, replace=True)
        self.y3 = train_Full3['attack_label']
        self.x3 = train_Full3.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full4 = train_Full.sample(n=N // 10, replace=True)
        self.y4 = train_Full4['attack_label']
        self.x4 = train_Full4.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full5 = train_Full.sample(n=N // 10, replace=True)
        self.y5 = train_Full5['attack_label']
        self.x5 = train_Full5.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full6 = train_Full.sample(n=N // 10, replace=True)
        self.y6 = train_Full6['attack_label']
        self.x6 = train_Full6.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full7 = train_Full.sample(n=N // 10, replace=True)
        self.y7 = train_Full7['attack_label']
        self.x7 = train_Full7.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full8 = train_Full.sample(n=N // 10, replace=True)
        self.y8 = train_Full8['attack_label']
        self.x8 = train_Full8.drop(["unknown", "attack", "attack_label"], axis=1)
        train_Full9 = train_Full.sample(n=N // 10, replace=True)
        self.y9 = train_Full9['attack_label']
        self.x9 = train_Full9.drop(["unknown", "attack", "attack_label"], axis=1)
        print('Create Classfiers:')

        # choose different machine learning algorithms as different classifiers
        # make classifier for Randomforest
        if num == 'R':
            #self. model() will train the model for each machine learning algorithm we picked
            clf_r = self.model('R')

            print('Randomforest')
            # score = clf_r.score(x, y)
            score2 = clf_r.score(x_test, y_test)
            # print(score)
            print(score2)
            # predict the test part to get the accuray of the final model
            y_train_pred = clf_r.predict(x_test)
            output = confusion_matrix(y_test, y_train_pred)
            # print the accuracy for the new model
            TN = output[0,0]
            TP = output[1,1]
            FP = output[1,0]
            FN = output[0,1]
            Accuracy = (TP+TN )/(TP+TN+FP+FN )
            Precision = (TP) / (TP + FP)
            FPR = (FP) / (FP + TN)
            F1_measure = (2*TP) / (2*TP+FP+FN)
            Recall = (TP) / (TP + FN)
            print('Accuracy='+ str(Accuracy))
            print('Precision='+str(Precision))
            print('Recall=' +str(Recall))
            print('FPR='+ str(FPR))
            print('F1_measure='+ str(F1_measure))
            return clf_r
        # return this model for Randomforest

        # make classifier for SVM
        if num == 'S':

            print("SVM")
            clf_s = self.model('S')
            #self. model() will train the model for each machine learning algorithm we picked
            y_train_pred = clf_s.predict(x_test)
            output = confusion_matrix(y_test, y_train_pred)
            TN = output[0,0]
            TP = output[1,1]
            FP = output[1,0]
            FN = output[0,1]
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            Precision = (TP) / (TP + FP)
            FPR = (FP) / (FP + TN)
            F1_measure = (2 * TP) / (2 * TP + FP + FN)
            Recall = (TP) / (TP + FN)
            print('Accuracy='+ str(Accuracy))
            print('Precision='+str(Precision))
            print('Recall=' +str(Recall))
            print('FPR='+ str(FPR))
            print('F1_measure='+ str(F1_measure))
            return clf_s
        # return this model for SVM

        if num == 'P':
            # train model for Pipeline
            print("Pipeline")
            clf_x = self.model('P')
            y_train_pred = clf_x.predict(x_test)
            output = confusion_matrix(y_test, y_train_pred)
            TN = output[0,0]
            TP = output[1,1]
            FP = output[1,0]
            FN = output[0,1]
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            Precision = (TP) / (TP + FP)
            FPR = (FP) / (FP + TN)
            F1_measure = (2 * TP) / (2 * TP + FP + FN)
            Recall = (TP) / (TP + FN)
            print('Accuracy=' + str(Accuracy))
            print('Precision=' + str(Precision))
            print('Recall=' + str(Recall))
            print('FPR=' + str(FPR))
            print('F1_measure=' + str(F1_measure))
            return clf_x
        # return this model

        if num == 'K':
            print("KNeighbors")
            clf_k = self.model('K')
            y_train_pred = clf_k.predict(x_test)
            output = confusion_matrix(y_test, y_train_pred)
            TN = output[0,0]
            TP = output[1,1]
            FP = output[1,0]
            FN = output[0,1]
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            Precision = (TP) / (TP + FP)
            FPR = (FP) / (FP + TN)
            F1_measure = (2 * TP) / (2 * TP + FP + FN)
            Recall = (TP) / (TP + FN)
            print('Accuracy='+ str(Accuracy))
            print('Precision='+str(Precision))
            print('Recall=' +str(Recall))
            print('FPR='+ str(FPR))
            print('F1_measure='+ str(F1_measure))
            return clf_k


        if num == 'D':
            clf_d = self.model('D')
            y_train_pred = clf_d.predict(x_test)
            output = confusion_matrix(y_test, y_train_pred)
            # print('output:')
            # print(output)
            TN = output[0, 0]
            TP = output[1, 1]
            FP = output[1, 0]
            FN = output[0, 1]
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            Precision = (TP) / (TP + FP)
            FPR = (FP) / (FP + TN)
            F1_measure = (2 * TP) / (2 * TP + FP + FN)
            Recall = (TP) / (TP + FN)
            print('Accuracy='+ str(Accuracy))
            print('Precision='+str(Precision))
            print('Recall=' +str(Recall))
            print('FPR='+ str(FPR))
            print('F1_measure='+ str(F1_measure))
            # score = clf_d.score(x, y)
            # score2 = clf_d.score(x_test, y_test)
            # y_train_pred = cross_val_predict(clf_d, x_test, y_test, cv=10)
            # output = confusion_matrix(y_test, y_train_pred)
            # print('output:')
            # print(output)
            if self.SAVE:
                # Generate tree
                dot_data = tree.export_graphviz(clf_d, out_file=None, feature_names=x_test.columns,
                                                class_names=["Benign", "Attack"])
                graph = graphviz.Source(dot_data)

                graph.render()
            # draw the bounary tree for the machine learning algorithm of decitrion tree
            return clf_d
        else:
            if num != 'D' and num!='R' and num !='S' and num !='P' and num !='K':
                clf_d =[]
#------------------------------------------------------------
    #     y_pred = clf_d.predict(x_test)
    #     conf_matrix = confusion_matrix(y_test, y_pred)
    #     prec_score = precision_score(y_test, y_pred)
    #     acc_score = accuracy_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
# -------------------------------------------------------------

        if self.DEBUG:
            print(num)
            print(self.num) # check the num of samples
            print(self.add) # check the attack samples of this type
            print(self.ATTACK) # check the attack seed for this algorithm

        return clf_d


    def model(self,num):

        # we start with selecting each method:

        print('10 fold cross-validation')
        if num == 'D':
            clf_d2 = tree.DecisionTreeClassifier(max_depth=5)
            print('For Decison Tree:')
        else:
            if num == 'R':
                clf_d2 = ensemble.RandomForestClassifier(n_estimators=30)
                print('For RandomForest:')
            if num == 'S':
                clf_d2 = svm.SVC(C=1000,kernel='linear')
                print('For SVM:')
            if num == 'P':
                clf_d2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
                print('P:')
            if num == 'K':
                clf_d2 = neighbors.KNeighborsClassifier()
                print('K:')
            else:
                clf_d2 = tree.DecisionTreeClassifier(max_depth=5)
                print('For Decison Tree:')

        score0 = 0
        clf_d2 = clf_d2.fit(self.x0, self.y0)
        clf_d20 = clf_d2
        score = clf_d20.score(self.x0, self.y0)
        # calculate the score of each 10 fold cross valdilations,
        # selected the top 1 as the final classifier for each machine learning methods
        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 1')
        # First model

        clf_d2 = clf_d2.fit(self.x1, self.y1)
        score = clf_d2.score(self.x1, self.y1)

        score2 = clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 2')
        # 2nd model

        clf_d2 = clf_d2.fit(self.x2, self.y2)
        score = clf_d2.score(self.x2,self.y2)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 3')
        # 3rd model


        clf_d2 = clf_d2.fit(self.x3, self.y3)
        score = clf_d2.score(self.x3,self. y3)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 4')
        # 4th model


        clf_d2 = clf_d2.fit(self.x4, self.y4)
        score = clf_d2.score(self.x4, self.y4)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x0,
                                                                                                   self.y0) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 5')
        # 5th model

        clf_d2 = clf_d2.fit(self.x5, self.y5)
        score = clf_d2.score(self.x5, self.y5)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x0, self.y0) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 6')
        # 6th model

        clf_d2 = clf_d2.fit(self.x6, self.y6)
        score = clf_d2.score(self.x6, self.y6)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 7')
        # 7th model

        clf_d2 = clf_d2.fit(self.x7,self.y7)
        score = clf_d2.score(self.x7, self.y7)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 8')
        # 8th model


        clf_d2 = clf_d2.fit(self.x8, self.y8)
        score = clf_d2.score(self.x8, self.y8)

        score2 = clf_d2.score(self.x1,self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x3, self.y3) + clf_d2.score(self.x9, self.y9)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 9')
        # 9th model

        clf_d2 = clf_d2.fit(self.x9, self.y9)
        score = clf_d2.score(self.x9, self.y9)

        score2 = clf_d2.score(self.x1, self.y1) + clf_d2.score(self.x2, self.y2) + clf_d2.score(self.x0, self.y0) + clf_d2.score(self.x4,
                                                                                                   self.y4) + clf_d2.score(
            self.x5, self.y5) + clf_d2.score(self.x6, self.y6) + clf_d2.score(self.x7, self.y7) + clf_d2.score(self.x8, self.y8) + clf_d2.score(self.x3, self.y3)
        score2 = score2 / 9
        if score2>score0:
            score0 = score2
            clf_df = clf_d2
        print('Bootstrapping Sample 10')
        # 10th model

        return clf_df
