####################################################################################################################################################
# AD_Prediction_by_DNN.py
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date Created: 11/02/2018
# input
#       1. gene expression (samples x genes) with label (AD, Normal)
#       2. DNA methylation (samples x CpG probes) with label (AD, Normal)
#       3. DEG list (Normal vs AD)
#       4. DMP list (Normal vs AD)
# output
#       prediction performance by DNN and various machine learning algorithms while varying dimension reduction (feature selection) algorithms
####################################################################################################################################################

import tensorflow as tf
import numpy as np
import os.path
import random as rd
import shutil as shu
import itertools
import csv
import pandas as pd

import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


graph_DNN7 = tf.Graph()
graph_DNN9 = tf.Graph()
graph_DNN11 = tf.Graph()
default_graph = tf.get_default_graph()


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


def saveDataToFile(filename, inputArr, featureList):
    print("saveDataToFile")
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except OSError:
            pass
    fout = open(filename, 'w')

    for i in range(len(featureList)):
        if(i == (len(featureList)-1)):
            fout.write(featureList[i] + "\n")
        else:
            fout.write(featureList[i] + "\t")

    for i in range(len(inputArr)):
        for j in range(len(inputArr[i])):
            if(j == (len(inputArr[i]-1))):
                fout.write(str(inputArr[i][j]) + "\n")
            else:
                fout.write(str(inputArr[i][j]) + "\t")

    print("complete to save the file")



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def buildIntegratedDataset_notinteg(xy_omics, mode):
    print("buildIntegratedDataset_notinteg")
    xy_omics_list = []
    n_row, n_col = xy_omics.shape
    print("xy_omics: " + xy_omics.shape.__str__())

    # build random index pair set
    idxSet_No = set()
    idxSet_AD = set()

    NoArr = [1., 0.]
    ADArr = [0., 1.]
    NoCnt = 0
    ADCnt = 0

    for idx in range(0, n_row - 1):
        label = xy_omics[idx][-2:]
        # normal
        if np.array_equal(label, NoArr):
            idxSet_No.add(idx)
            NoCnt += 1

        # AD
        if np.array_equal(label, ADArr):
            idxSet_AD.add(idx)
            ADCnt += 1

    balanced_sample_size = 0;
    if (NoCnt > ADCnt):
        balanced_sample_size = ADCnt

    if (NoCnt < ADCnt):
        balanced_sample_size = NoCnt

    print("NoCnt: " + NoCnt.__str__())
    print("ADCnt: " + ADCnt.__str__())
    print("size of idxSet_No: " + len(idxSet_No).__str__())
    print("size of idxSet_AD: " + len(idxSet_AD).__str__())

    NoCnt = 0
    ADCnt = 0
    if mode == "balanced":
        print("balanced_sample_size: " + balanced_sample_size.__str__())

        # for normal
        cnt = 0
        for idx in range(len(idxSet_No)):
            value = xy_omics[idx][:-2]

            xy_values_tmp = []
            xy_values_tmp.insert(0, str(idx))
            for i in range(len(value)):
                xy_values_tmp.insert(i + 1, value[i])
            xy_values_tmp.insert(len(xy_values_tmp) + 1, 0)
            xy_omics_list.append(xy_values_tmp)
            cnt += 1
            if (cnt >= balanced_sample_size):
                break

        # for AD
        cnt = 0
        for idx in range(len(idxSet_AD)):
            value = xy_omics[idx][:-2]

            xy_values_tmp = []
            xy_values_tmp.insert(0, str(idx))
            for i in range(len(value)):
                xy_values_tmp.insert(i + 1, value[i])
            xy_values_tmp.insert(len(xy_values_tmp) + 1, 1)
            xy_omics_list.append(xy_values_tmp)
            cnt += 1
            if (cnt >= balanced_sample_size):
                break

    if mode != "balanced":
        for idx in range(0, n_row - 1):
            value = xy_omics[idx][:-2]
            label = xy_omics[idx][-2:]

            xy_values_tmp = []
            xy_values_tmp.insert(0, str(idx))

            for i in range(len(value)):
                xy_values_tmp.insert(i + 1, value[i])

            # normal
            if np.array_equal(label, NoArr):
                xy_values_tmp.insert(len(xy_values_tmp) + 1, 0)
                NoCnt += 1

            # AD
            if np.array_equal(label, ADArr):
                xy_values_tmp.insert(len(xy_values_tmp) + 1, 1)
                ADCnt += 1

            xy_omics_list.append(xy_values_tmp)

    #print(xy_omics_list)
    print("xy_omics_list: " + len(xy_omics_list).__str__())
    xy_values = np.array(xy_omics_list)

    return xy_values


def buildIntegratedDataset_DNN(xy_gxpr, xy_meth, mode):
    print("buildIntegratedDataset")
    xy_gxpr_meth = []

    n_row_g, n_col_g = xy_gxpr.shape
    n_row_m, n_col_m = xy_meth.shape

    # build random index pair set
    idxSet_No = set()
    idxSet_AD = set()

    NoArr = [1., 0.]
    ADArr = [0., 1.]
    NoCnt = 0
    ADCnt = 0

    for idx_g in range(0, n_row_g - 1):
        label_g = xy_gxpr[idx_g][-2:]
        # print(label_g)
        for idx_m in range(0, n_row_m - 1):
            label_m = xy_meth[idx_m][-2:]
            # print(label_m)

            # normal
            if np.array_equal(label_g, NoArr) and np.array_equal(label_m, NoArr):
                integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
                idxSet_No.add(integ_idx)
                # print("normal: " + integ_idx)
                NoCnt += 1

            # AD
            if np.array_equal(label_g, ADArr) and np.array_equal(label_m, ADArr):
                integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
                idxSet_AD.add(integ_idx)
                # print("ad: " + integ_idx)
                ADCnt += 1

    print("NoCnt: " + NoCnt.__str__())
    print("ADCnt: " + ADCnt.__str__())
    print("size of idxSet_No: " + len(idxSet_No).__str__())
    print("size of idxSet_AD: " + len(idxSet_AD).__str__())

    balanced_sample_size = 0;
    if(len(idxSet_No) > len(idxSet_AD)):
        balanced_sample_size = len(idxSet_AD)

    if (len(idxSet_AD) > len(idxSet_No)):
        balanced_sample_size = len(idxSet_No)

    if mode == "balanced":
        print("balanced_sample_size: " + balanced_sample_size.__str__())

        # for normal
        cnt = 0
        for idx in range(len(idxSet_No)):
            idx_str = idxSet_No.pop()
            idx_str_split_list = idx_str.split('_')

            idx_ge_str = idx_str_split_list[0]
            idx_me_str = idx_str_split_list[1]
            idx_ge = int(idx_ge_str)
            idx_me = int(idx_me_str)

            value_ge = xy_gxpr[idx_ge][:-2]
            value_me = xy_meth[idx_me][:-2]

            xy_me_ge_values_tmp = []
            xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

            for i in range(len(value_me)):
                xy_me_ge_values_tmp.insert(i + 1, value_me[i])

            for j in range(len(value_ge)):
                xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

            #xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
            xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
            xy_gxpr_meth.append(xy_me_ge_values_tmp)

            cnt += 1
            if(cnt >= balanced_sample_size):
                break

        # for AD
        cnt = 0
        for idx in range(len(idxSet_AD)):
            idx_str = idxSet_AD.pop()
            idx_str_split_list = idx_str.split('_')

            idx_ge_str = idx_str_split_list[0]
            idx_me_str = idx_str_split_list[1]
            idx_ge = int(idx_ge_str)
            idx_me = int(idx_me_str)

            value_ge = xy_gxpr[idx_ge][:-2]
            value_me = xy_meth[idx_me][:-2]

            xy_me_ge_values_tmp = []
            xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

            for i in range(len(value_me)):
                xy_me_ge_values_tmp.insert(i + 1, value_me[i])

            for j in range(len(value_ge)):
                xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

            #xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
            xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
            xy_gxpr_meth.append(xy_me_ge_values_tmp)

            cnt += 1
            if (cnt >= balanced_sample_size):
                break

    if mode != "balanced":
        # for normal
        for idx in range(len(idxSet_No)):
            idx_str = idxSet_No.pop()
            idx_str_split_list = idx_str.split('_')

            idx_ge_str = idx_str_split_list[0]
            idx_me_str = idx_str_split_list[1]
            idx_ge = int(idx_ge_str)
            idx_me = int(idx_me_str)

            value_ge = xy_gxpr[idx_ge][:-2]
            value_me = xy_meth[idx_me][:-2]

            xy_me_ge_values_tmp = []
            xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

            for i in range(len(value_me)):
                xy_me_ge_values_tmp.insert(i + 1, value_me[i])

            for j in range(len(value_ge)):
                xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

            #xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
            xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
            xy_gxpr_meth.append(xy_me_ge_values_tmp)

        # for AD
        for idx in range(len(idxSet_AD)):
            idx_str = idxSet_AD.pop()
            idx_str_split_list = idx_str.split('_')

            idx_ge_str = idx_str_split_list[0]
            idx_me_str = idx_str_split_list[1]
            idx_ge = int(idx_ge_str)
            idx_me = int(idx_me_str)

            value_ge = xy_gxpr[idx_ge][:-2]
            value_me = xy_meth[idx_me][:-2]

            xy_me_ge_values_tmp = []
            xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

            for i in range(len(value_me)):
                xy_me_ge_values_tmp.insert(i + 1, value_me[i])

            for j in range(len(value_ge)):
                xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

            #xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
            xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
            xy_gxpr_meth.append(xy_me_ge_values_tmp)


    print("xy_gxpr_meth: " + len(xy_gxpr_meth).__str__())
    for idx in range(0, 10):
        xy = xy_gxpr_meth[idx]
        geneSet_str = ";".join(str(x) for x in xy)
        print(geneSet_str)

    xy_me_ge_values = np.array(xy_gxpr_meth)
    print(xy_me_ge_values.shape)

    return xy_me_ge_values



def getDEG_limma(filename, Thres_lfc, Thres_pval):
    geneSet = set()
    f = open(filename, 'r')
    inCSV = csv.reader(f, delimiter="\t")
    header = next(inCSV)  # for header

    for row in inCSV:
        gene = row[0]
        logFC = float(row[1])
        Pval = float(row[4]) ## adj p-val : row[5]

        if abs(logFC) >= Thres_lfc and Pval < Thres_pval:
            geneSet.add(gene)

    print("[limma] Number of gene set: " + str(len(geneSet)))

    return geneSet



def applyDimReduction_DEG_intersectGene(infilename, geneSet, filter_fn, Thres_lfc, Thres_pval):
    print("applyDimReduction_DEG_intersectGene")

    selected_genelist = ['SampleID']
    f = open(filter_fn, 'r')
    inCSV = csv.reader(f, delimiter="\t")
    header = next(inCSV)  # for header

    gene_fc_dict = {}
    for row in inCSV:
        gene = row[0]
        logFC = float(row[1])
        Pval = float(row[4])  ## adj p-val : row[5]

        if abs(logFC) >= Thres_lfc and Pval < Thres_pval:
            if gene in geneSet:
                gene_fc_dict[gene] = logFC


    sorted_gene_fc_list = sorted(gene_fc_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_gene_fc_list)

    cnt = 0
    for k_v in sorted_gene_fc_list:
        key = k_v[0]
        selected_genelist.append(key)

    # Label_No	Label_AD
    selected_genelist.append('Label_No')
    selected_genelist.append('Label_AD')

    print(str(len(selected_genelist)))
    print(selected_genelist)

    xy_all_df = pd.read_csv(infilename, sep='\t')
    xy_sel_df = xy_all_df[selected_genelist]
    xy = xy_sel_df.as_matrix()
    print("xy shape: " + xy.shape.__str__())

    xy_values = xy[1:, 1:-2]
    xy_labels = xy[1:, -2:]

    print(xy_values)
    print(xy_labels)

    # Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
    xy_labels_1_column = []

    NoArr = [1, 0]
    # AD array
    ADArr = [0, 1]
    num_rows, num_cols = xy_labels.shape
    for i in range(num_rows):
        if np.array_equal(xy_labels[i], NoArr):
            xy_labels_1_column.append(0)
        if np.array_equal(xy_labels[i], ADArr):
            xy_labels_1_column.append(1)

    print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
    print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

    X_embedded = xy_values
    print(X_embedded.shape)
    XY_embedded = np.append(X_embedded, xy_labels, axis=1)
    print(XY_embedded.shape)
    print(XY_embedded)
    return XY_embedded




def getDMG(filename):
    geneSet = set()
    cpgSet = set()

    f = open(filename, 'r')
    inCSV = csv.reader(f, delimiter="\t")
    header = next(inCSV)  # for header
    for row in inCSV:
        cpg = row[0]
        gene = row[6]
        cpgSet.add(cpg)
        geneSet.add(gene)

    print("Number of CpG set: " + str(len(cpgSet)))
    print("Number of gene set: " + str(len(geneSet)))
    return geneSet, cpgSet


def applyDimReduction_DMP_intersectGene(infilename, geneSet, filter_fn):
    print("applyDimReduction_DMP_intersectGene")
    selected_cpglist = ['SampleID']
    f = open(filter_fn, 'r')
    inCSV = csv.reader(f, delimiter="\t")

    for row in inCSV:
        cpg = row[0]
        gene = row[6]
        if gene in geneSet:
            selected_cpglist.append(cpg)

    # Label_No	Label_AD
    selected_cpglist.append('Label_No')
    selected_cpglist.append('Label_AD')

    print(str(len(selected_cpglist)))
    print(selected_cpglist)

    xy_all_df = pd.read_csv(infilename, sep='\t')
    xy_sel_df = xy_all_df[selected_cpglist]
    xy = xy_sel_df.as_matrix()
    print("xy shape: " + xy.shape.__str__())

    xy_tp = np.transpose(xy)
    print("xy_tp shape: " + xy_tp.shape.__str__())

    xy_values = xy[1:, 1:-2]
    xy_labels = xy[1:, -2:]

    print(xy_values)
    print(xy_labels)

    # Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
    xy_labels_1_column = []

    NoArr = [1, 0]
    # AD array
    ADArr = [0, 1]
    num_rows, num_cols = xy_labels.shape
    for i in range(num_rows):
        if np.array_equal(xy_labels[i], NoArr):
            xy_labels_1_column.append(0)
        if np.array_equal(xy_labels[i], ADArr):
            xy_labels_1_column.append(1)

    print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
    print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
    print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

    X_embedded = xy_values
    print(X_embedded.shape)
    XY_embedded = np.append(X_embedded, xy_labels, axis=1)
    print(XY_embedded.shape)
    print(XY_embedded)

    return XY_embedded


def applyDimReduction_PCA(infilename, num_comp, scatterPlot_fn):
    # https://indico.io/blog/visualizing-with-t-sne/
    print("applyDimReduction PCA")
    xy = np.genfromtxt(infilename, unpack=True, delimiter='\t', dtype=str)
    print("xy shape: " + xy.shape.__str__())
    xy_tp = np.transpose(xy)
    print("xy_tp shape: " + xy_tp.shape.__str__())

    xy_featureList = xy_tp[0, 1:]
    print("xy_featureList shape: " + xy_featureList.shape.__str__())
    print(xy_featureList)

    xy_values = xy_tp[1:, 1:-2]
    xy_labels = xy_tp[1:, -2:]

    xy_values = xy_values.astype(np.float)
    xy_labels = xy_labels.astype(np.float)

    print(xy_values)
    print(xy_labels)

    # Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
    xy_labels_1_column = []

    NoArr = [1, 0]
    # AD array
    ADArr = [0, 1]
    num_rows, num_cols = xy_labels.shape
    for i in range(num_rows):
        if np.array_equal(xy_labels[i], NoArr):
            xy_labels_1_column.append(0)
        if np.array_equal(xy_labels[i], ADArr):
            xy_labels_1_column.append(1)

    print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
    print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
    print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

    # print(xy_values)
    # print(xy_labels_1_column)
    # apply PCA
    pca = PCA(n_components=num_comp, svd_solver='full')
    pca.fit(xy_values)
    X_embedded = pca.transform(xy_values)
    print(X_embedded.shape)

    print(X_embedded)
    print(X_embedded.shape)
    XY_embedded = np.append(X_embedded, xy_labels, axis=1)
    print(XY_embedded)
    print(XY_embedded.shape)

    if (num_comp == 2):
        # before TSNE : randomly select row * 2 matrix
        n_row, n_col = xy_values.shape
        idx_x = rd.randint(0, n_col - 1)
        idx_y = rd.randint(0, n_col - 1)
        vis_x_before = xy_values[:, idx_x]
        vis_y_before = xy_values[:, idx_y]

        print("idx_x: " + idx_x.__str__() + "\t" + "idx_y: " + idx_y.__str__())

        plt.scatter(vis_x_before, vis_y_before, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(ticks=range(2))
        plt.clim(0, 1)
        #plt.show()
        fn = scatterPlot_fn + " [before].png"
        plt.savefig(fn, dpi=300)
        plt.close()

        # plot the result
        vis_x = X_embedded[:, 0]
        vis_y = X_embedded[:, 1]

        plt.scatter(vis_x, vis_y, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(ticks=range(2))
        plt.clim(0, 1)
        plt.show()
        fn = scatterPlot_fn + " [after].png"
        plt.savefig(fn, dpi=300)
        plt.close()

    if (num_comp == 3):
        # before TSNE : randomly select row * 2 matrix
        n_row, n_col = xy_values.shape
        idx_x = rd.randint(0, n_col - 1)
        idx_y = rd.randint(0, n_col - 1)
        idx_z = rd.randint(0, n_col - 1)
        vis_x_before = xy_values[:, idx_x]
        vis_y_before = xy_values[:, idx_y]
        vis_z_before = xy_values[:, idx_z]
        print("idx_x: " + idx_x.__str__() + "\t" + "idx_y: " + idx_y.__str__() + "\t" + "idx_z: " + idx_z.__str__())

        fig = plt.figure()
        ax = Axes3D(fig)
        line = ax.scatter(vis_x_before, vis_y_before, vis_z_before, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(line)
        # plt.clim(0, 1)
        fn = scatterPlot_fn + " [before].png"
        fig.savefig(fn, dpi=300)
        # plt.show()
        # ax.clear()
        plt.close()

        # plot the result
        vis_x = X_embedded[:, 0]
        vis_y = X_embedded[:, 1]
        vis_z = X_embedded[:, 2]

        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        line2 = ax2.scatter(vis_x, vis_y, vis_z, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(line2)
        # plt.clim(0, 1)
        fn = scatterPlot_fn + " [after].png"
        fig2.savefig(fn, dpi=300)
        # plt.show()
        # ax2.clear()
        plt.close()

    return XY_embedded


def applyDimReduction_TSNE(infilename, num_comp, scatterPlot_fn):
    print("applyDimReduction_TSNE")
    xy = np.genfromtxt(infilename, unpack=True, delimiter='\t', dtype=str)
    print("xy shape: " + xy.shape.__str__())
    xy_tp = np.transpose(xy)
    print("xy_tp shape: " + xy_tp.shape.__str__())

    xy_featureList = xy_tp[0, 1:]
    print("xy_featureList shape: " + xy_featureList.shape.__str__())
    print(xy_featureList)

    xy_values = xy_tp[1:, 1:-2]
    xy_labels = xy_tp[1:, -2:]

    xy_values = xy_values.astype(np.float)
    xy_labels = xy_labels.astype(np.float)

    # Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
    xy_labels_1_column = []

    NoArr = [1, 0]
    # AD array
    ADArr = [0, 1]
    num_rows, num_cols = xy_labels.shape
    for i in range(num_rows):
        if np.array_equal(xy_labels[i], NoArr):
            xy_labels_1_column.append(0)
        if np.array_equal(xy_labels[i], ADArr):
            xy_labels_1_column.append(1)

    print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
    print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
    print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

    #print(xy_values)
    #print(xy_labels_1_column)

    X_embedded = TSNE(n_components=num_comp).fit_transform(xy_values)
    print(X_embedded.shape)
    #print(X_embedded)

    XY_embedded = np.append(X_embedded, xy_labels, axis=1)
    print("XY_embedded: " + XY_embedded.shape.__str__())
    #print(XY_embedded)

    if(num_comp == 2):
        # before TSNE : randomly select row * 2 matrix
        n_row, n_col = xy_values.shape
        idx_x = rd.randint(0, n_col - 1)
        idx_y = rd.randint(0, n_col - 1)
        vis_x_before = xy_values[:, idx_x]
        vis_y_before = xy_values[:, idx_y]

        print("idx_x: " + idx_x.__str__() + "\t" + "idx_y: " + idx_y.__str__())

        plt.scatter(vis_x_before, vis_y_before, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(ticks=range(2))
        plt.clim(0, 1)
        plt.show()
        fn = scatterPlot_fn + " [before].png"
        plt.savefig(fn, dpi=300)
        plt.close()

        # plot the result
        vis_x = X_embedded[:, 0]
        vis_y = X_embedded[:, 1]

        plt.scatter(vis_x, vis_y, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(ticks=range(2))
        plt.clim(0, 1)
        plt.show()
        fn = scatterPlot_fn + " [after].png"
        plt.savefig(fn, dpi=300)
        plt.close()

    if (num_comp == 3):
        # before TSNE : randomly select row * 2 matrix
        n_row, n_col = xy_values.shape
        idx_x = rd.randint(0, n_col - 1)
        idx_y = rd.randint(0, n_col - 1)
        idx_z = rd.randint(0, n_col - 1)
        vis_x_before = xy_values[:, idx_x]
        vis_y_before = xy_values[:, idx_y]
        vis_z_before = xy_values[:, idx_z]
        print("idx_x: " + idx_x.__str__() + "\t" + "idx_y: " + idx_y.__str__() + "\t" + "idx_z: " + idx_z.__str__())

        fig = plt.figure()
        ax = Axes3D(fig)
        line = ax.scatter(vis_x_before, vis_y_before, vis_z_before, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(line)
        #plt.clim(0, 1)
        fn = scatterPlot_fn + " [before].png"
        fig.savefig(fn, dpi=300)
        #plt.show()
        #ax.clear()
        plt.close()

        # plot the result
        vis_x = X_embedded[:, 0]
        vis_y = X_embedded[:, 1]
        vis_z = X_embedded[:, 2]

        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        line2 = ax2.scatter(vis_x, vis_y, vis_z, c=xy_labels_1_column, cmap=plt.cm.get_cmap("RdBu"))
        #plt.colorbar(line2)
        #plt.clim(0, 1)
        fn = scatterPlot_fn + " [after].png"
        fig2.savefig(fn, dpi=300)
        #plt.show()
        #ax2.clear()
        plt.close()

    return XY_embedded


def partitionTrainTest_ML_for_CV(xy_me_ge_values):
    np.random.shuffle(xy_me_ge_values)

    x_data_List = []
    y_data_List = []

    colSize = len(xy_me_ge_values[0])

    for i in range(len(xy_me_ge_values)):
        x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
        y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]

        x_data_List.append(x_tmpRow)
        y_data_List.append(y_tmpRow)

    return np.array(x_data_List), np.array(y_data_List)


def partitionTrainTest_ML_for_CV_DNN(xy_me_ge_values):
    np.random.shuffle(xy_me_ge_values)

    x_data_List = []
    y_data_List = []

    colSize = len(xy_me_ge_values[0])

    for i in range(len(xy_me_ge_values)):
        x_tmpRow = xy_me_ge_values[i, 1:colSize - 2]
        y_tmpRow = xy_me_ge_values[i, colSize - 1:colSize]

        x_data_List.append(x_tmpRow)
        y_data_List.append(y_tmpRow)

    return np.array(x_data_List), np.array(y_data_List)


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score


def doMachineLearning_single(xy_me_ge_values, outfilename, mode, case):
    print("doMachineLearning_single")

    resultFilePath = "./" + case + "_"

    with open(outfilename, 'w') as fout:
        # fout = open(outfilename, 'w')
        fout.write("Do DNN with DNA methylation and Gene expression\n")

        train_test_ratio = 3 / 3
        rowSize = len(xy_me_ge_values)
        colSize = len(xy_me_ge_values[0])
        trainSize = int(rowSize * (train_test_ratio))

        print("rowSize: ", rowSize.__str__(), "\t" + "colSize: ", colSize.__str__())
        print("trainSize: ", trainSize.__str__())
        fout.write("rowSize: " + rowSize.__str__() + "\t" + "colSize: " + colSize.__str__() + "\n")
        fout.write("trainSize: " + trainSize.__str__() + "\n")

        print(xy_me_ge_values)

        # preprocess data scale
        if (mode.__eq__("ML and DNN")):
            print("ML and DNN mode")
            x_data, y_data = partitionTrainTest_ML_for_CV_DNN(xy_me_ge_values)
        if (mode.__eq__("Only ML")):
            print("ML mode")
            x_data, y_data = partitionTrainTest_ML_for_CV(xy_me_ge_values)

        sc = StandardScaler()
        sc.fit(x_data)
        x_data_std = sc.transform(x_data)

        # modify label information
        c, r = y_data.shape
        y_data = y_data.reshape(c, )

        # x_data_std = x_data

        # Random Forest ###############################################################################################
        # new test
        print("new Random forest")
        for i in range(0, 5):
            X_train, X_test, y_train, y_test = train_test_split(x_data_std, y_data, test_size=0.2)
            tr_cnt_no = 0
            tr_cnt_ad = 0
            te_cnt_no = 0
            te_cnt_ad = 0

            for val in y_train:
                if val == "1":
                    tr_cnt_ad += 1
                if val == "0":
                    tr_cnt_no += 1

            for val in y_test:
                if val == "1":
                    te_cnt_ad += 1
                if val == "0":
                    te_cnt_no += 1

            print("Train no: " + tr_cnt_no.__str__())
            print("Train AD: " + tr_cnt_ad.__str__())
            print("Test no: " + te_cnt_no.__str__())
            print("Test AD: " + te_cnt_ad.__str__())

            rf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1,
                                        random_state=0, max_depth=6)
            rf.fit(X_train, y_train)
            predicted = rf.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            print("Out-of-bag error: " + (1 - rf.oob_score_).__str__())
            print("Mean accuracy score: " + accuracy.__str__())
            print("confusion_matrix")
            print(confusion_matrix(y_test, predicted))

            class_names = ["0", "1"]
            cnf_matrix = confusion_matrix(y_test, predicted)
            np.set_printoptions(precision=2)
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix, without normalization')

            plt.savefig(resultFilePath + 'confusion_matrix_randomforest.pdf')
            plt.savefig(resultFilePath + 'confusion_matrix_randomforest.png')
            plt.close()

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix')
            plt.savefig(resultFilePath + 'confusion_matrix_norm_randomforest.pdf')
            plt.savefig(resultFilePath + 'confusion_matrix_norm_randomforest.png')
            plt.close()
            print("")

        plt.close()

        print("Random Forest")
        fout.write("\n\n")
        fout.write("Random Forest")

        ###############################################################################################
        # define a model #rdf_clf.fit(x_data_std, y_data)
        rdf_clf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1,
                                         random_state=0, max_depth=6)

        # croess validation
        #scores_k5 = cross_val_score(rdf_clf, x_data_std, y_data, cv=5, scoring='accuracy')
        scores_k5 = cross_val_score(rdf_clf, x_data_std, y_data, cv=5, scoring=make_scorer(classification_report_with_accuracy_score))
        print("5-fold CV accuracy for each fold")
        print(scores_k5)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2))
        #nested_score = cross_val_score(rdf_clf, x_data_std, y_data, cv=5,
        #                               scoring=make_scorer(classification_report_with_accuracy_score))
        #print(nested_score)

        fout.write("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2) + "\n")
        fout.write(scores_k5.__str__() + "\n")

        ###############################################################################################
        # ROC analysis with cross validation
        cv = StratifiedKFold(n_splits=5)
        classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=-1, random_state=0, max_depth=6)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(x_data_std, y_data):
            probas_ = classifier.fit(x_data_std[train], y_data[train]).predict_proba(x_data_std[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1], pos_label='1')
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC CV %d (AUC = %0.4f)' % (i, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='baseline', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for RandomForest')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(resultFilePath + 'Roc_curve_RandomForest.pdf')
        plt.savefig(resultFilePath + 'Roc_curve_RandomForest.png')
        plt.close()
        ###############################################################################################

        # SVM ###############################################################################################
        print("SVM")
        fout.write("\n\n")
        fout.write("SVM (with RBF)" + "\n")
        ###############################################################################################
        # define a model #rdf_clf.fit(x_data_std, y_data)
        svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                          max_iter=-1, probability=True, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)

        # croess validation
        scores_k5 = cross_val_score(svm_clf, x_data_std, y_data, cv=5, scoring=make_scorer(classification_report_with_accuracy_score))
        print("5-fold CV accuracy for each fold")
        print(scores_k5)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2))

        fout.write("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2) + "\n")
        fout.write(scores_k5.__str__() + "\n")


        ###############################################################################################
        # ROC analysis with cross validation
        cv = StratifiedKFold(n_splits=5)
        classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                             max_iter=-1, probability=True, random_state=None, shrinking=True,
                             tol=0.001, verbose=False)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(x_data_std, y_data):
            probas_ = classifier.fit(x_data_std[train], y_data[train]).predict_proba(x_data_std[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1], pos_label='1')
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC CV %d (AUC = %0.4f)' % (i, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='baseline', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for SVM')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(resultFilePath + 'Roc_curve_SVM.pdf')
        plt.savefig(resultFilePath + 'Roc_curve_SVM.png')
        plt.close()
        ###############################################################################################


        # naive bayesian classifier
        print("## naive bayesian")
        fout.write("\n\n")
        fout.write("naive bayesian" + "\n")
        ###############################################################################################
        # define a model #rdf_clf.fit(x_data_std, y_data)
        gnb_clf = GaussianNB()

        # croess validation
        scores_k5 = cross_val_score(gnb_clf, x_data_std, y_data, cv=5, scoring=make_scorer(classification_report_with_accuracy_score))
        print("5-fold CV accuracy for each fold")
        print(scores_k5)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2))
        fout.write("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2) + "\n")
        fout.write(scores_k5.__str__() + "\n")

        ###############################################################################################
        # ROC analysis with cross validation
        cv = StratifiedKFold(n_splits=5)
        classifier = GaussianNB()

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(x_data_std, y_data):
            probas_ = classifier.fit(x_data_std[train], y_data[train]).predict_proba(x_data_std[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1], pos_label='1')
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC CV %d (AUC = %0.4f)' % (i, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='baseline', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for Gaussian Naive Bayes')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(resultFilePath + 'Roc_curve_gaussian_naive_Bayes.pdf')
        plt.savefig(resultFilePath + 'Roc_curve_gaussian_naive_Bayes.png')
        plt.close()
        ###############################################################################################

    fout.close()


def doMachineLearning(xy_me_ge_values, outfilename, mode, case):
    print("doMachineLearning")

    resultFilePath = case + "_"

    with open(outfilename, 'w') as fout:
    #fout = open(outfilename, 'w')
        fout.write("Do DNN with DNA methylation and Gene expression\n")

        train_test_ratio = 3 / 3
        rowSize = len(xy_me_ge_values)
        colSize = len(xy_me_ge_values[0])
        trainSize = int(rowSize * (train_test_ratio))

        print("rowSize: ", rowSize.__str__(), "\t" + "colSize: ", colSize.__str__())
        print("trainSize: ", trainSize.__str__())
        fout.write("rowSize: " + rowSize.__str__() + "\t" + "colSize: " + colSize.__str__() + "\n")
        fout.write("trainSize: " + trainSize.__str__() + "\n")

        print(xy_me_ge_values)

        # preprocess data scale
        if (mode.__eq__("ML and DNN")):
            print("ML and DNN mode")
            x_data, y_data = partitionTrainTest_ML_for_CV_DNN(xy_me_ge_values)
        if (mode.__eq__("Only ML")):
            print("ML mode")
            x_data, y_data = partitionTrainTest_ML_for_CV(xy_me_ge_values)


        sc = StandardScaler()
        sc.fit(x_data)
        x_data_std = sc.transform(x_data)
    
        # modify label information
        c, r = y_data.shape
        y_data = y_data.reshape(c, )

        # Random Forest ###############################################################################################
        # new test
        print("new Random forest")
        for i in range(0, 5):
            X_train, X_test, y_train, y_test = train_test_split(x_data_std, y_data, test_size=0.2)
            tr_cnt_no = 0
            tr_cnt_ad = 0
            te_cnt_no = 0
            te_cnt_ad = 0

            for val in y_train:
                if val == "1":
                    tr_cnt_ad += 1
                if val == "0":
                    tr_cnt_no += 1

            for val in y_test:
                if val == "1":
                    te_cnt_ad += 1
                if val == "0":
                    te_cnt_no += 1

            print("Train no: " + tr_cnt_no.__str__())
            print("Train AD: " + tr_cnt_ad.__str__())
            print("Test no: " + te_cnt_no.__str__())
            print("Test AD: " + te_cnt_ad.__str__())

            rf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1, random_state=0, max_depth=6)
            rf.fit(X_train, y_train)
            predicted = rf.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            print("Out-of-bag error: " + (1 - rf.oob_score_).__str__())
            print("Mean accuracy score: " + accuracy.__str__())
            print("confusion_matrix")
            print(confusion_matrix(y_test, predicted))

            class_names = ["0", "1"]
            cnf_matrix = confusion_matrix(y_test, predicted)
            np.set_printoptions(precision=2)
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix, without normalization')

            plt.savefig(resultFilePath + 'confusion_matrix_randomforest.pdf')
            plt.savefig(resultFilePath + 'confusion_matrix_randomforest.png')
            plt.close()

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix')
            plt.savefig(resultFilePath + 'confusion_matrix_norm_randomforest.pdf')
            plt.savefig(resultFilePath + 'confusion_matrix_norm_randomforest.png')
            plt.close()
            print("")

        plt.close()

        print("Random Forest")
        fout.write("\n\n")
        fout.write("Random Forest")

        ###############################################################################################
        # define a model #rdf_clf.fit(x_data_std, y_data)
        rdf_clf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1, random_state=0, max_depth=6)

        # croess validation
        scores_k5 = cross_val_score(rdf_clf, x_data_std, y_data, cv=5, scoring=make_scorer(classification_report_with_accuracy_score))
        #scores_k5_oobscore = cross_val_score(rdf_clf, x_data_std, y_data, cv=5, scoring='oob_score')
        print("5-fold CV accuracy for each fold")
        print(scores_k5)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2))
        fout.write("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2) + "\n")
        fout.write(scores_k5.__str__() + "\n")

        # ROC analysis with cross validation
        cv = StratifiedKFold(n_splits=5)
        classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=-1, random_state=0, max_depth=6)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(x_data_std, y_data):
            probas_ = classifier.fit(x_data_std[train], y_data[train]).predict_proba(x_data_std[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1], pos_label='1')
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC CV %d (AUC = %0.4f)' % (i, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='baseline', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for RandomForest')
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig(resultFilePath + 'Roc_curve_RandomForest.pdf')
        plt.savefig(resultFilePath + 'Roc_curve_RandomForest.png')
        plt.close()
        ###############################################################################################

        # SVM ###############################################################################################
        print("SVM")
        fout.write("\n\n")
        fout.write("SVM (with RBF)" + "\n")
        ###############################################################################################
        # define a model #rdf_clf.fit(x_data_std, y_data)
        svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                          max_iter=-1, probability=True, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)

        # croess validation
        scores_k5 = cross_val_score(svm_clf, x_data_std, y_data, cv=5, scoring = make_scorer(classification_report_with_accuracy_score))
        print("5-fold CV accuracy for each fold")
        print(scores_k5)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2))
        fout.write("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2) + "\n")
        fout.write(scores_k5.__str__() + "\n")


        ###############################################################################################
        # ROC analysis with cross validation
        cv = StratifiedKFold(n_splits=5)
        classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                             max_iter=-1, probability=True, random_state=None, shrinking=True,
                             tol=0.001, verbose=False)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(x_data_std, y_data):
            probas_ = classifier.fit(x_data_std[train], y_data[train]).predict_proba(x_data_std[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1], pos_label='1')
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC CV %d (AUC = %0.4f)' % (i, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='baseline', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for SVM')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(resultFilePath + 'Roc_curve_SVM.pdf')
        plt.savefig(resultFilePath + 'Roc_curve_SVM.png')
        plt.close()
        ###############################################################################################


        # naive bayesian classifier
        print("## naive bayesian")
        fout.write("\n\n")
        fout.write("naive bayesian" + "\n")
        ###############################################################################################
        # define a model #rdf_clf.fit(x_data_std, y_data)
        gnb_clf = GaussianNB()

        # croess validation
        scores_k5 = cross_val_score(gnb_clf, x_data_std, y_data, cv=5, scoring=make_scorer(classification_report_with_accuracy_score))
        print("5-fold CV accuracy for each fold")
        print(scores_k5)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2))
        fout.write("Accuracy: %0.4f (+/- %0.4f)" % (scores_k5.mean(), scores_k5.std() * 2) + "\n")
        fout.write(scores_k5.__str__() + "\n")

        ###############################################################################################
        # ROC analysis with cross validation
        cv = StratifiedKFold(n_splits=5)
        classifier = GaussianNB()

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(x_data_std, y_data):
            probas_ = classifier.fit(x_data_std[train], y_data[train]).predict_proba(x_data_std[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_data[test], probas_[:, 1], pos_label='1')
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC CV %d (AUC = %0.4f)' % (i, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='baseline', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for Gaussian Naive Bayes')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(resultFilePath + 'Roc_curve_gaussian_naive_Bayes.pdf')
        plt.savefig(resultFilePath + 'Roc_curve_gaussian_naive_Bayes.png')
        plt.close()
        ###############################################################################################


    fout.close()




def partitionTrainTest_unbalanced(xy_me_ge_values, train_test_ratio):
    np.random.shuffle(xy_me_ge_values)
    # No array
    NoArr = ['0']
    # AD array
    ADArr = ['1']

    x_train_data_List = []
    y_train_data_List = []
    x_test_data_List = []
    y_test_data_List = []

    print("xy_me_ge_values size => row: ", len(xy_me_ge_values))
    print("xy_me_ge_values size [0] => col: ", len(xy_me_ge_values[0]))
    np.random.shuffle(xy_me_ge_values)

    rowSize = len(xy_me_ge_values)
    colSize = len(xy_me_ge_values[0])
    trainSize = int(rowSize * (train_test_ratio))
    testSize = rowSize - trainSize
    print("trainSize: ", trainSize.__str__(), "\t" + "testSize: ", testSize.__str__())

    trNoCnt = 0;
    trADCnt = 0;
    teNoCnt = 0;
    teADCnt = 0;

    for i in range(0, trainSize):
        label = xy_me_ge_values[i][colSize-1:colSize]

        # Normal - train
        if(np.array_equal(label, NoArr)):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_train_data_List.append(x_tmpRow)
            y_train_data_List.append(y_tmpRow)
            trNoCnt += 1

            # remove
            np.delete(x_tmpRow, np.s_[:])
            np.delete(y_tmpRow, np.s_[:])
            #print("normal-train : " + trNoCnt.__str__())

        # AD - train
        if (np.array_equal(label, ADArr)):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_train_data_List.append(x_tmpRow)
            y_train_data_List.append(y_tmpRow)
            trADCnt += 1

            # remove
            np.delete(x_tmpRow, np.s_[:])
            np.delete(y_tmpRow, np.s_[:])
            #print("AD-train : " + trADCnt.__str__())

        np.delete(label, np.s_[:])

    for i in range(0, testSize):
        label = xy_me_ge_values[i][colSize-1:colSize]

        # Normal - test
        if(np.array_equal(label, NoArr)):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_test_data_List.append(x_tmpRow)
            y_test_data_List.append(y_tmpRow)
            teNoCnt += 1

            # remove
            np.delete(x_tmpRow, np.s_[:])
            np.delete(y_tmpRow, np.s_[:])
            #print("normal-test : " + teNoCnt.__str__())

        # AD - test
        if (np.array_equal(label, ADArr)):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_test_data_List.append(x_tmpRow)
            y_test_data_List.append(y_tmpRow)
            teADCnt += 1

            # remove
            np.delete(x_tmpRow, np.s_[:])
            np.delete(y_tmpRow, np.s_[:])
            #print("AD-test : " + teADCnt.__str__())

        np.delete(label, np.s_[:])


    np.delete(xy_me_ge_values, np.s_[:])

    sampleInfo = "trainSize_No: " + trNoCnt.__str__() + "\t" + " trainSize_AD: " + trNoCnt.__str__() + " testSize_No: " + teNoCnt.__str__(), "\t" + " testSize_AD: " + teADCnt.__str__()
    print(sampleInfo)

    return np.array(x_train_data_List), np.array(y_train_data_List), np.array(x_test_data_List), np.array(y_test_data_List), sampleInfo


def partitionTrainTest_balanced(xy_me_ge_values, train_test_ratio):
    np.random.shuffle(xy_me_ge_values)
    # No array
    NoArr = ['0']
    # AD array
    ADArr = ['1']

    x_train_data_List = []
    y_train_data_List = []
    x_test_data_List = []
    y_test_data_List = []

    print("xy_me_ge_values size => row: ", len(xy_me_ge_values))
    print("xy_me_ge_values size [0] => col: ", len(xy_me_ge_values[0]))
    np.random.shuffle(xy_me_ge_values)

    rowSize = len(xy_me_ge_values)
    colSize = len(xy_me_ge_values[0])
    trainSize = int(rowSize * (train_test_ratio))
    testSize = rowSize - trainSize
    print("trainSize: ", trainSize.__str__(), "\t" + "testSize: ", testSize.__str__())

    trNoMax = trainSize/2
    trADMax = trainSize/2
    teNoMax = testSize/2
    teADMax = testSize/2

    trNoCnt = 0
    trADCnt = 0
    teNoCnt = 0
    teADCnt = 0

    idx_pool = []
    for i in range(0, len(xy_me_ge_values)):
        idx_pool.append(i)

    print(idx_pool)

    while idx_pool:
        i = idx_pool.pop()
        label = xy_me_ge_values[i][colSize - 1:colSize]

        # Normal - train
        if(np.array_equal(label, NoArr) and trNoCnt <= trNoMax):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_train_data_List.append(x_tmpRow)
            y_train_data_List.append(y_tmpRow)
            trNoCnt += 1

        # AD - train
        if (np.array_equal(label, ADArr) and trADCnt <= trADMax):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_train_data_List.append(x_tmpRow)
            y_train_data_List.append(y_tmpRow)
            trADCnt += 1

        # Normal - test
        if(np.array_equal(label, NoArr) and teNoCnt <= teNoMax):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_test_data_List.append(x_tmpRow)
            y_test_data_List.append(y_tmpRow)
            teNoCnt += 1

        # AD - test
        if (np.array_equal(label, ADArr) and teADCnt <= teADMax):
            x_tmpRow = xy_me_ge_values[i, 1:colSize-1]
            y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
            x_test_data_List.append(x_tmpRow)
            y_test_data_List.append(y_tmpRow)
            teADCnt += 1


    np.delete(xy_me_ge_values, np.s_[:])

    sampleInfo = "trainSize_No: " + trNoCnt.__str__() + "\t" + " trainSize_AD: " + trADCnt.__str__() + " testSize_No: " + teNoCnt.__str__(), "\t" + " testSize_AD: " + teADCnt.__str__()
    print(sampleInfo)


    return np.array(x_train_data_List), np.array(y_train_data_List), np.array(x_test_data_List), np.array(y_test_data_List), sampleInfo




def doDNN_7(xy_me_ge_values, outfilename, mode, drop_out_rate, total_epoch):
    with graph_DNN7.as_default():
        print("doDNN_7")
        print(xy_me_ge_values)
        fout = open(outfilename, 'w')
        fout.write("Do DNN with DNA methylation and Gene expression\n")
        print("xy_me_ge_values size => row: ", len(xy_me_ge_values))
        print("xy_me_ge_values size [0] => col: ", len(xy_me_ge_values[0]))

        train_test_ratio = 4/5

        rowSize = len(xy_me_ge_values)
        colSize = len(xy_me_ge_values[0])
        trainSize = int(rowSize * (train_test_ratio))

        #x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest(xy_me_ge_values, train_test_ratio)
        if mode == "unbalanced":
            x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest_unbalanced(xy_me_ge_values, train_test_ratio)
        if mode == "balanced":
            x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest_balanced(xy_me_ge_values, train_test_ratio)

        x_train_data = x_train_data.astype(np.float)
        x_test_data = x_test_data.astype(np.float)

        y_train_data = y_train_data.astype(np.int)
        y_test_data = y_test_data.astype(np.int)

        NoArr = [0]
        ADArr = [1]
        tr_no_cnt = 0
        tr_ad_cnt = 0
        te_no_cnt = 0
        te_ad_cnt = 0

        for label in y_train_data:
            if np.array_equal(label, NoArr):
                tr_no_cnt += 1
            if np.array_equal(label, ADArr):
                tr_ad_cnt += 1

        for label in y_test_data:
            if np.array_equal(label, NoArr):
                te_no_cnt += 1
            if np.array_equal(label, ADArr):
                te_ad_cnt += 1

        print("training dataset No: " + tr_no_cnt.__str__() + "\t" + "AD: " + tr_ad_cnt.__str__())
        print("testing dataset No: " + te_no_cnt.__str__() + "\t" + "AD: " + te_ad_cnt.__str__())
        print("[before] trainSize: " + trainSize.__str__())
        trainSize = tr_no_cnt + tr_ad_cnt
        print("[after] trainSize: " + trainSize.__str__())
        print("colSize: " + colSize.__str__())

        nSize_L1 = 300
        nSize_L2 = 300
        nSize_L3 = 300
        nSize_L4 = 300
        nSize_L5 = 300
        nSize_L6 = 300
        nSize_L7 = 300
        nSize_L8 = 2

        ###################################
        print(x_train_data.shape, y_train_data.shape)
        print(x_test_data.shape, y_test_data.shape)

        classes = 2 # number of class label : 0 ~ 1

        X = tf.placeholder(tf.float32, shape=[None, colSize-2])  # 19448 features
        Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 labels [0 or 1]

        Y_one_hot = tf.one_hot(Y, classes)  # one-hot : rank가 1 증가됨
        print('one-hot :', Y_one_hot)
        Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # reshape로 shape 변환
        print('reshape :', Y_one_hot)

        ###################################
        W1_dnn8 = tf.get_variable("W1_dnn8", shape=[colSize - 2, nSize_L1], initializer=xavier_init(colSize - 2, nSize_L1))
        W2_dnn8 = tf.get_variable("W2_dnn8", shape=[nSize_L1, nSize_L2], initializer=xavier_init(nSize_L1, nSize_L2))
        W3_dnn8 = tf.get_variable("W3_dnn8", shape=[nSize_L2, nSize_L3], initializer=xavier_init(nSize_L2, nSize_L3))
        W4_dnn8 = tf.get_variable("W4_dnn8", shape=[nSize_L3, nSize_L4], initializer=xavier_init(nSize_L3, nSize_L4))
        W5_dnn8 = tf.get_variable("W5_dnn8", shape=[nSize_L4, nSize_L5], initializer=xavier_init(nSize_L4, nSize_L5))
        W6_dnn8 = tf.get_variable("W6_dnn8", shape=[nSize_L5, nSize_L6], initializer=xavier_init(nSize_L5, nSize_L6))
        W7_dnn8 = tf.get_variable("W7_dnn8", shape=[nSize_L6, nSize_L7], initializer=xavier_init(nSize_L6, nSize_L7))
        W8_dnn8 = tf.get_variable("W8_dnn8", shape=[nSize_L7, nSize_L8], initializer=xavier_init(nSize_L7, nSize_L8))

        b1_dnn8 = tf.Variable(tf.random_normal([nSize_L1]), name="Bias1_dnn8")
        b2_dnn8 = tf.Variable(tf.random_normal([nSize_L2]), name="Bias2_dnn8")
        b3_dnn8 = tf.Variable(tf.random_normal([nSize_L3]), name="Bias3_dnn8")
        b4_dnn8 = tf.Variable(tf.random_normal([nSize_L4]), name="Bias4_dnn8")
        b5_dnn8 = tf.Variable(tf.random_normal([nSize_L5]), name="Bias5_dnn8")
        b6_dnn8 = tf.Variable(tf.random_normal([nSize_L6]), name="Bias6_dnn8")
        b7_dnn8 = tf.Variable(tf.random_normal([nSize_L7]), name="Bias7_dnn8")
        b8_dnn8 = tf.Variable(tf.random_normal([nSize_L8]), name="Bias8_dnn8")

        dropout_rate_dnn8 = tf.placeholder("float")  # sigmoid or relu

        Layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X, W1_dnn8), b1_dnn8)), dropout_rate_dnn8)
        Layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer1, W2_dnn8), b2_dnn8)), dropout_rate_dnn8)
        Layer3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer2, W3_dnn8), b3_dnn8)), dropout_rate_dnn8)
        Layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer3, W4_dnn8), b4_dnn8)), dropout_rate_dnn8)
        Layer5 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer4, W5_dnn8), b5_dnn8)), dropout_rate_dnn8)
        Layer6 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer5, W6_dnn8), b6_dnn8)), dropout_rate_dnn8)
        Layer7 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer6, W7_dnn8), b7_dnn8)), dropout_rate_dnn8)
        Layer8 = tf.add(tf.matmul(Layer7, W8_dnn8), b8_dnn8)

        # define logit, hypothesis
        logits = Layer8
        hypothesis = tf.nn.softmax(logits)

        # cross entropy cost function
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
        cost = tf.reduce_mean(cost_i)
        ##########################################################

        # Minimize error using cross entropy
        learning_rate = 0.1
        # Gradient Descent
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        prediction = tf.argmax(hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        training_epochs = total_epoch#13020  # 12000  # 1000
        dr_rate = float(drop_out_rate)

        ##############################################
        print("DNN structure")
        print("Relu / Softmax / Onehot vector encoding")
        print("nSize_L1: " + nSize_L1.__str__())
        print("nSize_L2: " + nSize_L2.__str__())
        print("nSize_L3: " + nSize_L3.__str__())
        print("nSize_L4: " + nSize_L4.__str__())
        print("nSize_L5: " + nSize_L5.__str__())
        print("nSize_L6: " + nSize_L6.__str__())
        print("nSize_L7: " + nSize_L7.__str__())
        print("nSize_L8: " + nSize_L8.__str__())
        print("DNN parameters")
        print("learning_rate: {:1.5f}".format(learning_rate))
        print("training_epochs: " + training_epochs.__str__())
        print("dr_rate: " + dr_rate.__str__())
        ##############################################

        # fout all information
        fout.write("DNN structure\n")
        fout.write("Relu / Softmax / Onehot vector encoding\n")
        fout.write("sampleInfo: " + sinfo.__str__() + "\n")
        fout.write("trainSize: " + trainSize.__str__() + "\n")
        fout.write("nSize_L1: " + nSize_L1.__str__() + "\n")
        fout.write("nSize_L2: " + nSize_L2.__str__() + "\n")
        fout.write("nSize_L3: " + nSize_L3.__str__() + "\n")
        fout.write("nSize_L4: " + nSize_L4.__str__() + "\n")
        fout.write("nSize_L5: " + nSize_L5.__str__() + "\n")
        fout.write("nSize_L6: " + nSize_L6.__str__() + "\n")
        fout.write("nSize_L7: " + nSize_L7.__str__() + "\n")
        fout.write("nSize_L8: " + nSize_L8.__str__() + "\n")
        fout.write("DNN parameters" + "\n")
        fout.write("learning_rate: {:1.5f}".format(learning_rate) + "\n")
        fout.write("training_epochs: " + training_epochs.__str__() + "\n")
        fout.write("dr_rate: " + dr_rate.__str__() + "\n")
        ##############################################

        # Launch the graph
        with tf.Session(graph=graph_DNN7) as sess_dnn8:
            sess_dnn8.run(tf.global_variables_initializer())

            for epoch in range(training_epochs):
                sess_dnn8.run(train, feed_dict={X: x_train_data, Y: y_train_data, dropout_rate_dnn8: dr_rate})

                if epoch % 10 == 0 or epoch == training_epochs-1:
                    loss, acc = sess_dnn8.run([cost, accuracy],
                                         feed_dict={X: x_train_data, Y: y_train_data, dropout_rate_dnn8: dr_rate})
                    print("epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(epoch, loss, acc))
                    trResult = "epoch:\t" + epoch.__str__() + "\t" + "cost:\t" + loss.__str__() + "\t" + "Training accuracy:\t" + acc.__str__()
                    fout.write(trResult + "\n")

            print("@@ Optimization Finished: sess.run with test data @@")
            pred = sess_dnn8.run(prediction, feed_dict={X: x_test_data, dropout_rate_dnn8: 1.0})

            no_cnt_true = 0
            ad_cnt_true = 0
            for y in y_test_data:
                if y == 0:
                    no_cnt_true += 1
                else:
                    ad_cnt_true += 1

            # AD: positive condition, No: negative condition
            tp_cnt = 0
            fp_cnt = 0
            fn_cnt = 0
            tn_cnt = 0
            print(pred)
            print("pred len: " + len(pred).__str__())

            for p, y in zip(pred, y_test_data.flatten()):
                # print(p.__str__() + "\t" + y.__str__())
                if p == int(y):
                    if (y == 1):
                        tp_cnt += 1
                    else:
                        tn_cnt += 1

                if p != int(y):
                    if (y == 1):
                        fn_cnt += 1
                    else:
                        fp_cnt += 1

            print("AD cnt: " + ad_cnt_true.__str__() + "\t" + "No cnt: " + no_cnt_true.__str__())
            print("TP: " + tp_cnt.__str__() + "\t" + "FP: " + fp_cnt.__str__())
            print("FN: " + fn_cnt.__str__() + "\t" + "TN: " + tn_cnt.__str__())
            # print(sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0}))

            print("@@ 1 Optimization Finished with accuracy.eval @@")
            teResult = accuracy.eval({X: x_test_data, Y: y_test_data, dropout_rate_dnn8: 1.0})
            print("Test Acc: ", teResult)
            fout.write("Test Accuracy:\t" + teResult.__str__() + '\n')

            print("@@ 2 Optimization Finished with accuracy.eval @@")
            loss, acc = sess_dnn8.run([cost, accuracy], feed_dict={X: x_test_data, Y: y_test_data, dropout_rate_dnn8: 1.0})
            print("Loss: {:.3f}\tAcc: {:.3%}".format(loss, acc))

            pred_arr = np.asarray(pred)
            print("@@ 3 ROC, AUC")
            print("y_test_data size: " + len(y_test_data).__str__() + "\t" + "pred_arr size: " + len(pred_arr).__str__())
            auc, update_op = tf.metrics.auc(y_test_data, pred_arr)
            print("AUC: " + auc.__str__())
            print("update_op: " + update_op.__str__())

            # by using sklearn
            auc_by_sk = roc_auc_score(y_test_data, pred_arr)
            print("auc_by_sk: " + auc_by_sk.__str__())
            print("precision: " + precision_score(y_test_data, pred_arr).__str__())
            print("recall: " + recall_score(y_test_data, pred_arr).__str__())
            print("f1_score: " + f1_score(y_test_data, pred_arr).__str__())
            print(confusion_matrix(y_test_data, pred_arr).__str__())
            fpr, tpr, thresholds = roc_curve(y_test_data, pred_arr)
            print("FPR")
            print(fpr.__str__())
            print("TPR")
            print(tpr.__str__())
            print("Thresholds")
            print(thresholds.__str__())

            sess_dnn8.close()

        fout.close()



def doDNN_9(xy_me_ge_values, outfilename, mode, drop_out_rate, total_epoch, determine_variable_reuse):
    with graph_DNN9.as_default():
        print("doDNN_9")
        print(xy_me_ge_values)
        fout = open(outfilename, 'w')
        fout.write("Do DNN with DNA methylation and Gene expression\n")
        print("xy_me_ge_values size => row: ", len(xy_me_ge_values))
        print("xy_me_ge_values size [0] => col: ", len(xy_me_ge_values[0]))

        train_test_ratio = 4/5

        rowSize = len(xy_me_ge_values)
        colSize = len(xy_me_ge_values[0])
        trainSize = int(rowSize * (train_test_ratio))

        if mode == "unbalanced":
            x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest_unbalanced(xy_me_ge_values, train_test_ratio)
        if mode == "balanced":
            x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest_balanced(xy_me_ge_values, train_test_ratio)

        print("before astype")
        print(x_train_data)
        print(y_train_data)

        x_train_data = x_train_data.astype(np.float)
        x_test_data = x_test_data.astype(np.float)
        y_train_data = y_train_data.astype(np.int)
        y_test_data = y_test_data.astype(np.int)

        print("after astype")
        print("used training dataset")
        print(x_train_data)
        print(y_train_data)

        NoArr = [0]
        ADArr = [1]
        tr_no_cnt = 0
        tr_ad_cnt = 0
        te_no_cnt = 0
        te_ad_cnt = 0

        for label in y_train_data:
            if np.array_equal(label, NoArr):
                tr_no_cnt += 1
            if np.array_equal(label, ADArr):
                tr_ad_cnt += 1

        for label in y_test_data:
            if np.array_equal(label, NoArr):
                te_no_cnt += 1
            if np.array_equal(label, ADArr):
                te_ad_cnt += 1

        print("training dataset No: " + tr_no_cnt.__str__() + "\t" + "AD: " + tr_ad_cnt.__str__())
        print("testing dataset No: " + te_no_cnt.__str__() + "\t" + "AD: " + te_ad_cnt.__str__())
        print("[before] trainSize: " + trainSize.__str__())
        trainSize = tr_no_cnt + tr_ad_cnt
        print("[after] trainSize: " + trainSize.__str__())
        print("colSize: " + colSize.__str__())

        nSize_L1 = 300
        nSize_L2 = 300
        nSize_L3 = 300
        nSize_L4 = 300
        nSize_L5 = 300
        nSize_L6 = 300
        nSize_L7 = 300
        nSize_L8 = 300
        nSize_L9 = 300
        nSize_L10 = 2

        ###################################
        print(x_train_data.shape, y_train_data.shape)
        print(x_test_data.shape, y_test_data.shape)
        classes = 2 # number of class label : 0 ~ 1

        ###################################
        if ("yes" in determine_variable_reuse):
            with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
                print("reuse true")

                X = tf.placeholder(tf.float32, shape=[None, colSize - 2])  # 19448 features
                Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 labels [0 or 1]
                Y_one_hot = tf.one_hot(Y, classes)  # one-hot : rank가 1 증가됨
                print('one-hot :', Y_one_hot)
                Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # reshape로 shape 변환
                print('reshape :', Y_one_hot)

                W1 = tf.get_variable("W1_ge", shape=[colSize - 2, nSize_L1], initializer=xavier_init(colSize - 2, nSize_L1))
                W2 = tf.get_variable("W2_ge", shape=[nSize_L1, nSize_L2], initializer=xavier_init(nSize_L1, nSize_L2))
                W3 = tf.get_variable("W3_ge", shape=[nSize_L2, nSize_L3], initializer=xavier_init(nSize_L2, nSize_L3))
                W4 = tf.get_variable("W4_ge", shape=[nSize_L3, nSize_L4], initializer=xavier_init(nSize_L3, nSize_L4))
                W5 = tf.get_variable("W5_ge", shape=[nSize_L4, nSize_L5], initializer=xavier_init(nSize_L4, nSize_L5))
                W6 = tf.get_variable("W6_ge", shape=[nSize_L5, nSize_L6], initializer=xavier_init(nSize_L5, nSize_L6))
                W7 = tf.get_variable("W7_ge", shape=[nSize_L6, nSize_L7], initializer=xavier_init(nSize_L6, nSize_L7))
                W8 = tf.get_variable("W8_ge", shape=[nSize_L7, nSize_L8], initializer=xavier_init(nSize_L7, nSize_L8))
                W9 = tf.get_variable("W9_ge", shape=[nSize_L8, nSize_L9], initializer=xavier_init(nSize_L8, nSize_L9))
                W10 = tf.get_variable("W10_ge", shape=[nSize_L9, nSize_L10], initializer=xavier_init(nSize_L9, nSize_L10))

                b1 = tf.Variable(tf.random_normal([nSize_L1]), name="Bias1")
                b2 = tf.Variable(tf.random_normal([nSize_L2]), name="Bias2")
                b3 = tf.Variable(tf.random_normal([nSize_L3]), name="Bias3")
                b4 = tf.Variable(tf.random_normal([nSize_L4]), name="Bias4")
                b5 = tf.Variable(tf.random_normal([nSize_L5]), name="Bias5")
                b6 = tf.Variable(tf.random_normal([nSize_L6]), name="Bias6")
                b7 = tf.Variable(tf.random_normal([nSize_L7]), name="Bias7")
                b8 = tf.Variable(tf.random_normal([nSize_L8]), name="Bias8")
                b9 = tf.Variable(tf.random_normal([nSize_L9]), name="Bias9")
                b10 = tf.Variable(tf.random_normal([nSize_L10]), name="Bias10")

                dropout_rate = tf.placeholder("float")  # sigmoid or relu


        if ("no" in determine_variable_reuse):
            print("first use")

            X = tf.placeholder(tf.float32, shape=[None, colSize - 2])  # 19448 features
            Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 labels [0 or 1]
            Y_one_hot = tf.one_hot(Y, classes)  # one-hot : rank가 1 증가됨
            print('one-hot :', Y_one_hot)
            Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # reshape로 shape 변환
            print('reshape :', Y_one_hot)

            W1 = tf.get_variable("W1_ge", shape=[colSize - 2, nSize_L1], initializer=xavier_init(colSize - 2, nSize_L1))
            W2 = tf.get_variable("W2_ge", shape=[nSize_L1, nSize_L2], initializer=xavier_init(nSize_L1, nSize_L2))
            W3 = tf.get_variable("W3_ge", shape=[nSize_L2, nSize_L3], initializer=xavier_init(nSize_L2, nSize_L3))
            W4 = tf.get_variable("W4_ge", shape=[nSize_L3, nSize_L4], initializer=xavier_init(nSize_L3, nSize_L4))
            W5 = tf.get_variable("W5_ge", shape=[nSize_L4, nSize_L5], initializer=xavier_init(nSize_L4, nSize_L5))
            W6 = tf.get_variable("W6_ge", shape=[nSize_L5, nSize_L6], initializer=xavier_init(nSize_L5, nSize_L6))
            W7 = tf.get_variable("W7_ge", shape=[nSize_L6, nSize_L7], initializer=xavier_init(nSize_L6, nSize_L7))
            W8 = tf.get_variable("W8_ge", shape=[nSize_L7, nSize_L8], initializer=xavier_init(nSize_L7, nSize_L8))
            W9 = tf.get_variable("W9_ge", shape=[nSize_L8, nSize_L9], initializer=xavier_init(nSize_L8, nSize_L9))
            W10 = tf.get_variable("W10_ge", shape=[nSize_L9, nSize_L10], initializer=xavier_init(nSize_L9, nSize_L10))

            b1 = tf.Variable(tf.random_normal([nSize_L1]), name="Bias1")
            b2 = tf.Variable(tf.random_normal([nSize_L2]), name="Bias2")
            b3 = tf.Variable(tf.random_normal([nSize_L3]), name="Bias3")
            b4 = tf.Variable(tf.random_normal([nSize_L4]), name="Bias4")
            b5 = tf.Variable(tf.random_normal([nSize_L5]), name="Bias5")
            b6 = tf.Variable(tf.random_normal([nSize_L6]), name="Bias6")
            b7 = tf.Variable(tf.random_normal([nSize_L7]), name="Bias7")
            b8 = tf.Variable(tf.random_normal([nSize_L8]), name="Bias8")
            b9 = tf.Variable(tf.random_normal([nSize_L9]), name="Bias9")
            b10 = tf.Variable(tf.random_normal([nSize_L10]), name="Bias10")

            dropout_rate = tf.placeholder("float")  # sigmoid or relu

        Layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X, W1), b1)), dropout_rate)
        Layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer1, W2), b2)), dropout_rate)
        Layer3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer2, W3), b3)), dropout_rate)
        Layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer3, W4), b4)), dropout_rate)
        Layer5 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer4, W5), b5)), dropout_rate)
        Layer6 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer5, W6), b6)), dropout_rate)
        Layer7 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer6, W7), b7)), dropout_rate)
        Layer8 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer7, W8), b8)), dropout_rate)
        Layer9 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer8, W9), b9)), dropout_rate)
        Layer10 = tf.add(tf.matmul(Layer9, W10), b10)

        # define logit, hypothesis
        logits = Layer10
        hypothesis = tf.nn.softmax(logits)

        # cross entropy cost function
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
        cost = tf.reduce_mean(cost_i)
        ##########################################################

        # Minimize error using cross entropy
        learning_rate = 0.1

        # Gradient Descent optimizer
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        prediction = tf.argmax(hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        training_epochs = total_epoch #13020  # 12000

        ##############################################
        print("DNN structure")
        print("Relu / Softmax / Onehot vector encoding")
        print("nSize_L1: " + nSize_L1.__str__())
        print("nSize_L2: " + nSize_L2.__str__())
        print("nSize_L3: " + nSize_L3.__str__())
        print("nSize_L4: " + nSize_L4.__str__())
        print("nSize_L5: " + nSize_L5.__str__())
        print("nSize_L6: " + nSize_L6.__str__())
        print("nSize_L7: " + nSize_L7.__str__())
        print("nSize_L8: " + nSize_L8.__str__())
        print("nSize_L9: " + nSize_L9.__str__())
        print("nSize_L10: " + nSize_L10.__str__())

        print("DNN parameters")
        print("learning_rate: {:1.5f}".format(learning_rate))
        print("training_epochs: " + training_epochs.__str__())
        print("dr_rate: " + drop_out_rate.__str__())
        ##############################################

        # fout all information
        fout.write("DNN structure\n")
        fout.write("Relu / Softmax / Onehot vector encoding\n")
        fout.write("sampleInfo: " + sinfo.__str__() + "\n")
        fout.write("trainSize: " + trainSize.__str__() + "\n")
        fout.write("nSize_L1: " + nSize_L1.__str__() + "\n")
        fout.write("nSize_L2: " + nSize_L2.__str__() + "\n")
        fout.write("nSize_L3: " + nSize_L3.__str__() + "\n")
        fout.write("nSize_L4: " + nSize_L4.__str__() + "\n")
        fout.write("nSize_L5: " + nSize_L5.__str__() + "\n")
        fout.write("nSize_L6: " + nSize_L6.__str__() + "\n")
        fout.write("nSize_L7: " + nSize_L7.__str__() + "\n")
        fout.write("nSize_L8: " + nSize_L8.__str__() + "\n")
        fout.write("nSize_L9: " + nSize_L9.__str__() + "\n")
        fout.write("nSize_L10: " + nSize_L10.__str__() + "\n")
        fout.write("DNN parameters" + "\n")
        fout.write("learning_rate: {:1.5f}".format(learning_rate) + "\n")
        fout.write("training_epochs: " + training_epochs.__str__() + "\n")
        fout.write("dr_rate: " + drop_out_rate.__str__() + "\n")
        ##############################################

        # Launch the graph
        with tf.Session(graph=graph_DNN9) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(training_epochs):
                sess.run(train, feed_dict={X: x_train_data, Y: y_train_data, dropout_rate: drop_out_rate})

                if epoch % 10 == 0 or epoch == training_epochs-1:
                    loss, acc = sess.run([cost, accuracy], feed_dict={X: x_train_data, Y: y_train_data, dropout_rate: drop_out_rate})
                    print("epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(epoch, loss, acc))

                    trResult = "epoch:\t" + epoch.__str__() + "\t" + "cost:\t" + loss.__str__() + "\t" + "Training accuracy:\t" + acc.__str__()
                    fout.write(trResult + "\n")

            print("@@ Optimization Finished: sess.run with test data @@")
            pred = sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0})

            no_cnt_true = 0
            ad_cnt_true = 0
            for y in y_test_data:
                if y == 0:
                    no_cnt_true += 1
                else:
                    ad_cnt_true += 1

            # AD: positive conditon, No: negative condition
            tp_cnt = 0
            fp_cnt = 0
            fn_cnt = 0
            tn_cnt = 0
            #print(pred)
            #print("pred len: " + len(pred).__str__())

            for p, y in zip(pred, y_test_data.flatten()):
                # print(p.__str__() + "\t" + y.__str__())
                if p == int(y):
                    if (y == 1):
                        tp_cnt += 1
                    else:
                        tn_cnt += 1

                if p != int(y):
                    if (y == 1):
                        fn_cnt += 1
                    else:
                        fp_cnt += 1

            print("AD cnt: " + ad_cnt_true.__str__() + "\t" + "No cnt: " + no_cnt_true.__str__())
            print("TP: " + tp_cnt.__str__() + "\t" + "FP: " + fp_cnt.__str__())
            print("FN: " + fn_cnt.__str__() + "\t" + "TN: " + tn_cnt.__str__())
            # print(sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0}))

            print("@@ 1 Optimization Finished with accuracy.eval @@")
            teResult = accuracy.eval({X: x_test_data, Y: y_test_data, dropout_rate: 1.0})
            print("Test Acc: ", teResult)
            fout.write("Test Accuracy:\t" + teResult.__str__() + '\n')

            print("@@ 2 Optimization Finished with accuracy.eval @@")
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_test_data, Y: y_test_data, dropout_rate: 1.0})
            print("Loss: {:.3f}\tAcc: {:.3%}".format(loss, acc))

            pred_arr = np.asarray(pred)
            print("@@ 3 ROC, AUC")
            print("y_test_data size: " + len(y_test_data).__str__() + "\t" + "pred_arr size: " + len(pred_arr).__str__())
            auc, update_op = tf.metrics.auc(y_test_data, pred_arr)
            print("AUC: " + auc.__str__())
            print("update_op: " + update_op.__str__())

            # by using sklearn
            auc_by_sk = roc_auc_score(y_test_data, pred_arr)
            print("auc_by_sk: " + auc_by_sk.__str__())
            print("precision: " + precision_score(y_test_data, pred_arr).__str__())
            print("recall: " + recall_score(y_test_data, pred_arr).__str__())
            print("f1_score: " + f1_score(y_test_data, pred_arr).__str__())
            print(confusion_matrix(y_test_data, pred_arr).__str__())
            fpr, tpr, thresholds = roc_curve(y_test_data, pred_arr)
            print("FPR")
            print(fpr.__str__())
            print("TPR")
            print(tpr.__str__())
            print("Thresholds")
            print(thresholds.__str__())

            sess.close()

        fout.close()



def doDNN_11(xy_me_ge_values, outfilename, mode, drop_out_rate, total_epoch):
    with graph_DNN11.as_default():
        print("doDNN_11")
        print(xy_me_ge_values)
        fout = open(outfilename, 'w')
        fout.write("Do DNN with DNA methylation and Gene expression\n")
        print("xy_me_ge_values size => row: ", len(xy_me_ge_values))
        print("xy_me_ge_values size [0] => col: ", len(xy_me_ge_values[0]))

        train_test_ratio = 4/5

        rowSize = len(xy_me_ge_values)
        colSize = len(xy_me_ge_values[0])
        trainSize = int(rowSize * (train_test_ratio))

        if mode == "unbalanced":
            x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest_unbalanced(xy_me_ge_values, train_test_ratio)
        if mode == "balanced":
            x_train_data, y_train_data, x_test_data, y_test_data, sinfo = partitionTrainTest_balanced(xy_me_ge_values, train_test_ratio)

        x_train_data = x_train_data.astype(np.float)
        x_test_data = x_test_data.astype(np.float)

        y_train_data = y_train_data.astype(np.int)
        y_test_data = y_test_data.astype(np.int)

        NoArr = [0]
        ADArr = [1]
        tr_no_cnt = 0
        tr_ad_cnt = 0
        te_no_cnt = 0
        te_ad_cnt = 0

        for label in y_train_data:
            if np.array_equal(label, NoArr):
                tr_no_cnt += 1
            if np.array_equal(label, ADArr):
                tr_ad_cnt += 1

        for label in y_test_data:
            if np.array_equal(label, NoArr):
                te_no_cnt += 1
            if np.array_equal(label, ADArr):
                te_ad_cnt += 1

        print("training dataset No: " + tr_no_cnt.__str__() + "\t" + "AD: " + tr_ad_cnt.__str__())
        print("testing dataset No: " + te_no_cnt.__str__() + "\t" + "AD: " + te_ad_cnt.__str__())
        print("[before] trainSize: " + trainSize.__str__())
        trainSize = tr_no_cnt + tr_ad_cnt
        print("[after] trainSize: " + trainSize.__str__())
        print("colSize: " + colSize.__str__())

        nSize_L1 = 300
        nSize_L2 = 300
        nSize_L3 = 300
        nSize_L4 = 300
        nSize_L5 = 300
        nSize_L6 = 300
        nSize_L7 = 300
        nSize_L8 = 300
        nSize_L9 = 300
        nSize_L10 = 300
        nSize_L11 = 300
        nSize_L12 = 2

        print(x_train_data.shape, y_train_data.shape)
        print(x_test_data.shape, y_test_data.shape)

        classes = 2 # number of class label : 0 ~ 1
        X = tf.placeholder(tf.float32, shape=[None, colSize-2])  # 19448 features
        Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 labels [0 or 1]
        Y_one_hot = tf.one_hot(Y, classes)  # one-hot : rank가 1 증가됨
        print('one-hot :', Y_one_hot)
        Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # reshape로 shape 변환
        print('reshape :', Y_one_hot)


        ###################################
        W1_dnn12 = tf.get_variable("W1_dnn12", shape=[colSize - 2, nSize_L1], initializer=xavier_init(colSize - 2, nSize_L1))
        W2_dnn12 = tf.get_variable("W2_dnn12", shape=[nSize_L1, nSize_L2], initializer=xavier_init(nSize_L1, nSize_L2))
        W3_dnn12 = tf.get_variable("W3_dnn12", shape=[nSize_L2, nSize_L3], initializer=xavier_init(nSize_L2, nSize_L3))
        W4_dnn12 = tf.get_variable("W4_dnn12", shape=[nSize_L3, nSize_L4], initializer=xavier_init(nSize_L3, nSize_L4))
        W5_dnn12 = tf.get_variable("W5_dnn12", shape=[nSize_L4, nSize_L5], initializer=xavier_init(nSize_L4, nSize_L5))
        W6_dnn12 = tf.get_variable("W6_dnn12", shape=[nSize_L5, nSize_L6], initializer=xavier_init(nSize_L5, nSize_L6))
        W7_dnn12 = tf.get_variable("W7_dnn12", shape=[nSize_L6, nSize_L7], initializer=xavier_init(nSize_L6, nSize_L7))
        W8_dnn12 = tf.get_variable("W8_dnn12", shape=[nSize_L7, nSize_L8], initializer=xavier_init(nSize_L7, nSize_L8))
        W9_dnn12 = tf.get_variable("W9_dnn12", shape=[nSize_L8, nSize_L9], initializer=xavier_init(nSize_L8, nSize_L9))
        W10_dnn12 = tf.get_variable("W10_dnn12", shape=[nSize_L9, nSize_L10], initializer=xavier_init(nSize_L9, nSize_L10))
        W11_dnn12 = tf.get_variable("W11_dnn12", shape=[nSize_L10, nSize_L11], initializer=xavier_init(nSize_L10, nSize_L11))
        W12_dnn12 = tf.get_variable("W12_dnn12", shape=[nSize_L11, nSize_L12], initializer=xavier_init(nSize_L11, nSize_L12))

        b1_dnn12 = tf.Variable(tf.random_normal([nSize_L1]), name="Bias1_dnn12")
        b2_dnn12 = tf.Variable(tf.random_normal([nSize_L2]), name="Bias2_dnn12")
        b3_dnn12 = tf.Variable(tf.random_normal([nSize_L3]), name="Bias3_dnn12")
        b4_dnn12 = tf.Variable(tf.random_normal([nSize_L4]), name="Bias4_dnn12")
        b5_dnn12 = tf.Variable(tf.random_normal([nSize_L5]), name="Bias5_dnn12")
        b6_dnn12 = tf.Variable(tf.random_normal([nSize_L6]), name="Bias6_dnn12")
        b7_dnn12 = tf.Variable(tf.random_normal([nSize_L7]), name="Bias7_dnn12")
        b8_dnn12 = tf.Variable(tf.random_normal([nSize_L8]), name="Bias8_dnn12")
        b9_dnn12 = tf.Variable(tf.random_normal([nSize_L9]), name="Bias9_dnn12")
        b10_dnn12 = tf.Variable(tf.random_normal([nSize_L10]), name="Bias10_dnn12")
        b11_dnn12 = tf.Variable(tf.random_normal([nSize_L11]), name="Bias11_dnn12")
        b12_dnn12 = tf.Variable(tf.random_normal([nSize_L12]), name="Bias12_dnn12")

        dropout_rate = tf.placeholder("float")  # sigmoid or relu

        Layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X, W1_dnn12), b1_dnn12)), dropout_rate)
        Layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer1, W2_dnn12), b2_dnn12)), dropout_rate)
        Layer3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer2, W3_dnn12), b3_dnn12)), dropout_rate)
        Layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer3, W4_dnn12), b4_dnn12)), dropout_rate)
        Layer5 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer4, W5_dnn12), b5_dnn12)), dropout_rate)
        Layer6 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer5, W6_dnn12), b6_dnn12)), dropout_rate)
        Layer7 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer6, W7_dnn12), b7_dnn12)), dropout_rate)
        Layer8 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer7, W8_dnn12), b8_dnn12)), dropout_rate)
        Layer9 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer8, W9_dnn12), b9_dnn12)), dropout_rate)
        Layer10 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer9, W10_dnn12), b10_dnn12)), dropout_rate)
        Layer11 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer10, W11_dnn12), b11_dnn12)), dropout_rate)
        Layer12 = tf.add(tf.matmul(Layer11, W12_dnn12), b12_dnn12)

        # define logit, hypothesis
        logits = Layer12
        hypothesis = tf.nn.softmax(logits)

        # cross entropy cost function
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
        cost = tf.reduce_mean(cost_i)

        # Minimize error using cross entropy
        learning_rate = 0.1
        # Gradient Descent
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        prediction = tf.argmax(hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        training_epochs = total_epoch#13020  # 12000  # 6000
        dr_rate = float(drop_out_rate)

        ##############################################
        print("DNN structure")
        print("Relu / Softmax / Onehot vector encoding")
        print("nSize_L1: " + nSize_L1.__str__())
        print("nSize_L2: " + nSize_L2.__str__())
        print("nSize_L3: " + nSize_L3.__str__())
        print("nSize_L4: " + nSize_L4.__str__())
        print("nSize_L5: " + nSize_L5.__str__())
        print("nSize_L6: " + nSize_L6.__str__())
        print("nSize_L7: " + nSize_L7.__str__())
        print("nSize_L8: " + nSize_L8.__str__())
        print("nSize_L9: " + nSize_L9.__str__())
        print("nSize_L10: " + nSize_L10.__str__())
        print("nSize_L11: " + nSize_L11.__str__())
        print("nSize_L12: " + nSize_L12.__str__())

        print("DNN parameters")
        print("learning_rate: {:1.5f}".format(learning_rate))
        print("training_epochs: " + training_epochs.__str__())
        print("dr_rate: " + dr_rate.__str__())
        ##############################################

        # fout all information
        fout.write("DNN structure\n")
        fout.write("Relu / Softmax / Onehot vector encoding\n")
        fout.write("sampleInfo: " + sinfo.__str__() + "\n")
        fout.write("trainSize: " + trainSize.__str__() + "\n")
        fout.write("nSize_L1: " + nSize_L1.__str__() + "\n")
        fout.write("nSize_L2: " + nSize_L2.__str__() + "\n")
        fout.write("nSize_L3: " + nSize_L3.__str__() + "\n")
        fout.write("nSize_L4: " + nSize_L4.__str__() + "\n")
        fout.write("nSize_L5: " + nSize_L5.__str__() + "\n")
        fout.write("nSize_L6: " + nSize_L6.__str__() + "\n")
        fout.write("nSize_L7: " + nSize_L7.__str__() + "\n")
        fout.write("nSize_L8: " + nSize_L8.__str__() + "\n")
        fout.write("nSize_L9: " + nSize_L9.__str__() + "\n")
        fout.write("nSize_L10: " + nSize_L10.__str__() + "\n")
        fout.write("nSize_L11: " + nSize_L11.__str__() + "\n")
        fout.write("nSize_L12: " + nSize_L12.__str__() + "\n")
        fout.write("DNN parameters" + "\n")
        fout.write("learning_rate: {:1.5f}".format(learning_rate) + "\n")
        fout.write("training_epochs: " + training_epochs.__str__() + "\n")
        fout.write("dr_rate: " + dr_rate.__str__() + "\n")
        ##############################################

        # Launch the graph
        with tf.Session(graph=graph_DNN11) as sess_dnn12:
            sess_dnn12.run(tf.global_variables_initializer())

            for epoch in range(training_epochs):
                sess_dnn12.run(train, feed_dict={X: x_train_data, Y: y_train_data, dropout_rate: dr_rate})

                if epoch % 10 == 0 or epoch == training_epochs-1:
                    loss, acc = sess_dnn12.run([cost, accuracy],
                                         feed_dict={X: x_train_data, Y: y_train_data, dropout_rate: dr_rate})
                    print("epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(epoch, loss, acc))
                    trResult = "epoch:\t" + epoch.__str__() + "\t" + "cost:\t" + loss.__str__() + "\t" + "Training accuracy:\t" + acc.__str__()
                    fout.write(trResult + "\n")

            print("@@ Optimization Finished: sess.run with test data @@")
            pred = sess_dnn12.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0})

            no_cnt_true = 0
            ad_cnt_true = 0
            for y in y_test_data:
                if y == 0:
                    no_cnt_true += 1
                else:
                    ad_cnt_true += 1

            # AD: positive conditon, No: negative condition
            tp_cnt = 0
            fp_cnt = 0
            fn_cnt = 0
            tn_cnt = 0
            print(pred)
            print("pred len: " + len(pred).__str__())

            for p, y in zip(pred, y_test_data.flatten()):
                # print(p.__str__() + "\t" + y.__str__())
                if p == int(y):
                    if (y == 1):
                        tp_cnt += 1
                    else:
                        tn_cnt += 1

                if p != int(y):
                    if (y == 1):
                        fn_cnt += 1
                    else:
                        fp_cnt += 1

            print("AD cnt: " + ad_cnt_true.__str__() + "\t" + "No cnt: " + no_cnt_true.__str__())
            print("TP: " + tp_cnt.__str__() + "\t" + "FP: " + fp_cnt.__str__())
            print("FN: " + fn_cnt.__str__() + "\t" + "TN: " + tn_cnt.__str__())

            print("@@ 1 Optimization Finished with accuracy.eval @@")
            teResult = accuracy.eval({X: x_test_data, Y: y_test_data, dropout_rate: 1.0})
            print("Test Acc: ", teResult)
            fout.write("Test Accuracy:\t" + teResult.__str__() + '\n')

            print("@@ 2 Optimization Finished with accuracy.eval @@")
            loss, acc = sess_dnn12.run([cost, accuracy], feed_dict={X: x_test_data, Y: y_test_data, dropout_rate: 1.0})
            print("Loss: {:.3f}\tAcc: {:.3%}".format(loss, acc))

            pred_arr = np.asarray(pred)
            print("@@ 3 ROC, AUC")
            print("y_test_data size: " + len(y_test_data).__str__() + "\t" + "pred_arr size: " + len(pred_arr).__str__())
            auc, update_op = tf.metrics.auc(y_test_data, pred_arr)
            print("AUC: " + auc.__str__())
            print("update_op: " + update_op.__str__())

            # by using sklearn
            auc_by_sk = roc_auc_score(y_test_data, pred_arr)
            print("auc_by_sk: " + auc_by_sk.__str__())
            print("precision: " + precision_score(y_test_data, pred_arr).__str__())
            print("recall: " + recall_score(y_test_data, pred_arr).__str__())
            print("f1_score: " + f1_score(y_test_data, pred_arr).__str__())
            print(confusion_matrix(y_test_data, pred_arr).__str__())
            fpr, tpr, thresholds = roc_curve(y_test_data, pred_arr)
            print("FPR")
            print(fpr.__str__())
            print("TPR")
            print(tpr.__str__())
            print("Thresholds")
            print(thresholds.__str__())

            sess_dnn12.close()

        fout.close()


def main():
    print("Gene Expression and DNA Methylation analysis for classification")

    if os.path.isfile('DNN_result_allforDNN_all.txt'):
        try:
            os.remove('DNN_result_allforDNN_all.txt')
        except OSError as ex:
            print(ex)
            pass

    if os.path.isdir('./logs/GE_Meth_Comb_logs'):
        try:
            shu.rmtree('./logs/GE_Meth_Comb_logs')
        except OSError as ex:
            print(ex)

    print(os.getcwd())

    dirPath = "./results"
    if not os.path.exists(dirPath): os.mkdir(dirPath)

    dirPath = "./results/table_1"
    if not os.path.exists(dirPath): os.mkdir(dirPath)

    dirPath = "./results/table_2"
    if not os.path.exists(dirPath): os.mkdir(dirPath)

    dirPath = "./results/table_3"
    if not os.path.exists(dirPath): os.mkdir(dirPath)

    dirPath = "./results/table_4_DNN"
    if not os.path.exists(dirPath): os.mkdir(dirPath)

    ## make directory
    dirPath_table1_pca = "./results/table_1/PCA"
    if not os.path.exists(dirPath_table1_pca): os.mkdir(dirPath_table1_pca)
    dirPath_table1_tsne = "./results/table_1/tsne"
    if not os.path.exists(dirPath_table1_tsne): os.mkdir(dirPath_table1_tsne)
    dirPath_table1_deg = "./results/table_1/deg"
    if not os.path.exists(dirPath_table1_deg): os.mkdir(dirPath_table1_deg)

    dirPath_table2_pca = "./results/table_2/PCA"
    if not os.path.exists(dirPath_table2_pca): os.mkdir(dirPath_table2_pca)
    dirPath_table2_tsne = "./results/table_2/tsne"
    if not os.path.exists(dirPath_table2_tsne): os.mkdir(dirPath_table2_tsne)
    dirPath_table2_dmg = "./results/table_2/dmg"
    if not os.path.exists(dirPath_table2_dmg): os.mkdir(dirPath_table2_dmg)

    dirPath_table3_pca = "./results/table_3/PCA"
    if not os.path.exists(dirPath_table3_pca): os.mkdir(dirPath_table3_pca)
    dirPath_table3_tsne = "./results/table_3/tsne"
    if not os.path.exists(dirPath_table3_tsne): os.mkdir(dirPath_table3_tsne)
    dirPath_table3_deg_dmg = "./results/table_3/deg_dmg"
    if not os.path.exists(dirPath_table3_deg_dmg): os.mkdir(dirPath_table3_deg_dmg)

    ## comment for replaced DEG
    # comparison test
    num_of_dim_gxpr = 3
    num_of_dim_meth = 3
    XY_dr_gxpr_tsne = applyDimReduction_TSNE("./dataset/allforDNN_ge.txt", num_of_dim_gxpr, dirPath_table1_tsne + "/tsne_scatter_plot_gxpr")
    XY_dr_gxpr_tsne_ml = buildIntegratedDataset_notinteg(XY_dr_gxpr_tsne, "balanced")
    ## gene expression only - tSNE - RF, SVM
    doMachineLearning_single(XY_dr_gxpr_tsne_ml, dirPath_table1_tsne + "/ML_result.txt", "Only ML", dirPath_table1_tsne + "/gxpr_tsne")

    XY_dr_meth_tsne = applyDimReduction_TSNE("./dataset/allforDNN_me.txt", num_of_dim_meth, dirPath_table2_tsne + "/tsne_scatter_plot_meth")
    XY_dr_meth_tsne_ml = buildIntegratedDataset_notinteg(XY_dr_meth_tsne, "balanced")
    ## Methylation only - tSNE - RF, SVM
    doMachineLearning_single(XY_dr_meth_tsne_ml, dirPath_table2_tsne + "/ML_result.txt", "Only ML", dirPath_table2_tsne + "/meth_tsne")

    XY_dr_gxpr_pca = applyDimReduction_PCA("./dataset/allforDNN_ge.txt", num_of_dim_gxpr, dirPath_table1_pca + "/PCA_scatter_plot_gxpr")
    XY_dr_gxpr_pca_ml = buildIntegratedDataset_notinteg(XY_dr_gxpr_pca, "balanced")
    ## gene expression only - PCA - RF, SVM
    doMachineLearning_single(XY_dr_gxpr_pca_ml, dirPath_table1_pca + "/ML_result.txt", "Only ML", dirPath_table1_pca + "/gxpr_pca")

    XY_dr_meth_pca = applyDimReduction_PCA("./dataset/allforDNN_me.txt", num_of_dim_meth, dirPath_table2_pca + "/PCA_scatter_plot_meth")
    XY_dr_meth_pca_ml = buildIntegratedDataset_notinteg(XY_dr_meth_pca, "balanced")
    ## Methylation only - PCA - RF, SVM
    doMachineLearning_single(XY_dr_meth_pca_ml, dirPath_table2_pca + "/ML_result.txt", "Only ML", dirPath_table2_pca + "/meth_pca")

    XY_dr_gxpr_meth_tsne = buildIntegratedDataset_DNN(XY_dr_gxpr_tsne, XY_dr_meth_tsne, "balanced")
    doMachineLearning(XY_dr_gxpr_meth_tsne, dirPath_table3_tsne + "/[ge_me_tsne] ML_result.txt", "Only ML",
                      dirPath_table3_tsne + "/[ge_me_tsne]")

    XY_dr_gxpr_meth_pca = buildIntegratedDataset_DNN(XY_dr_gxpr_pca, XY_dr_meth_pca, "balanced")
    doMachineLearning(XY_dr_gxpr_meth_pca, dirPath_table3_pca + "/[ge_me_pca] ML_result.txt", "Only ML",
                      dirPath_table3_pca + "/[ge_me_pca]")

    ##############################################################################
    ## our feature selection
    # 1 = 2 fold
    # 0.58 = 1.5 fold
    Thres_lfc = 1
    Thres_pval = 0.01
    degSet = getDEG_limma("./dataset/DEG_list.tsv", Thres_lfc, Thres_pval)
    dmgSet, cpgSet = getDMG("./dataset/DMP_list.tsv")
    its_geneSet = degSet & dmgSet
    print("its_geneSet: " + str(len(its_geneSet)))

    ## intersecting genes
    XY_dr_gxpr = applyDimReduction_DEG_intersectGene("./dataset/allforDNN_ge.txt", its_geneSet, "./dataset/DEG_list.tsv", Thres_lfc, Thres_pval)
    XY_dr_meth = applyDimReduction_DMP_intersectGene("./dataset/allforDNN_me.txt", its_geneSet, "./dataset/DMP_list.tsv")

    XY_dr_gxpr_ml = buildIntegratedDataset_notinteg(XY_dr_gxpr, "balanced")
    XY_dr_meth_ml = buildIntegratedDataset_notinteg(XY_dr_meth, "balanced")
    doMachineLearning_single(XY_dr_gxpr_ml, dirPath_table1_deg + "/[DEG] ML_result.txt", "Only ML", dirPath_table1_deg + "/[DEG]")
    doMachineLearning_single(XY_dr_meth_ml, dirPath_table2_dmg + "/[DMG] ML_result.txt", "Only ML", dirPath_table2_dmg + "/[DMG]")

    XY_dr_gxpr_meth = buildIntegratedDataset_DNN(XY_dr_gxpr, XY_dr_meth, "balanced")
    doMachineLearning(XY_dr_gxpr_meth, dirPath_table3_deg_dmg + "/[DEG_DMG_intersection] ML_result.txt", "Only ML", dirPath_table3_deg_dmg + "/[DEG_DMG_intersection]")

    ## do DNN
    doDNN_7(XY_dr_gxpr_meth, "./results/table_4_DNN/[DEG_DMG_intersection] DNN_layer7.txt", "balanced", 0.7, 4000)
    doDNN_9(XY_dr_gxpr_meth, "./results/table_4_DNN/[DEG_DMG_intersection] DNN_layer9.txt", "balanced", 0.7, 4000, "no")
    doDNN_11(XY_dr_gxpr_meth, "./results/table_4_DNN/[DEG_DMG_intersection] DNN_layer11.txt", "balanced", 0.7, 4000)



if __name__ == '__main__':
    main()
