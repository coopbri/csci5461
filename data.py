import numpy as np
import pandas as pd
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# ======================================================
# PCA dimensionality reduction and other data preparation
# ======================================================
def prepare_data(data, target, n, training_size, test_size, class_labels):
    
    # Split the data for training and testing
    sample_train, sample_test, label_train, label_test = train_test_split(data, target, test_size=0.3, random_state=12)
    
    # Standarize for Gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # Pick training size number of samples from each distribution
    training_input = {key: (sample_train[label_train == k, :])[:training_size] for k, key in enumerate(class_labels)}
    test_input = {key: (sample_train[label_train == k, :])[training_size:(
        training_size+test_size)] for k, key in enumerate(class_labels)}

    return sample_train, label_train, training_input, test_input


# ======================================================
# Breast Cancer Wisconsin
# ======================================================
def breast_cancer(training_size, test_size, n, PLOT_DATA):
    class_labels = [r'Benign', r'Malignant']
    
    df = pd.read_csv('datasets/breast-cancer-wisconsin.csv', header=None)
    df = df.replace({'B':0, 'M':1})
    data = df.iloc[:,2:]
    target = df.iloc[:,1]
    
    sample_train, label_train, training_input, test_input = prepare_data(data, target, n, training_size, test_size, class_labels)

    if PLOT_DATA:
        for k in range(0, 2):
            label = 'Benign' if k is 0 else 'Malignant'
            
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size],
                        label=label)
        
        plt.title("Breast Cancer Dataset")
        plt.legend()
        plt.show()

    return sample_train, training_input, test_input, class_labels


# ======================================================
# Ecoli
# ======================================================
def ecoli(training_size, test_size, n, PLOT_DATA):
    class_labels = [r'cp', r'im', r'pp', r'imU', r'om', r'omL', r'imL', r'imS']
    
    df = pd.read_csv('datasets/ecoli.csv', header=None)
    df = df.replace({'cp':0, 'im':1, 'pp':2, 'imU':3, 'om':4, 'omL':5, 'imL':6, 'imS':7})
    data = df.iloc[:,1:8]
    target = df.iloc[:,8]
    
    sample_train, label_train, training_input, test_input = prepare_data(data, target, n, training_size, test_size, class_labels)
    
    if PLOT_DATA:
        for k in range(0,8):
            if k == 0:
                label = 'cp'
            elif k == 1:
                label = 'im'
            elif k == 2:
                label = 'pp'
            elif k == 3:
                label = 'imU'
            elif k == 4:
                label = 'om'
            elif k == 5:
                label = 'omL'
            elif k == 6:
                label = 'imL'
            else:
                label = 'imS'
                
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size],
                        label=label)
        
        plt.title("E. Coli Dataset")
        plt.legend()
        plt.show()
        
    return sample_train, training_input, test_input, class_labels


# ======================================================          
# Yeast
# ======================================================
def yeast(training_size, test_size, n, PLOT_DATA):
    class_labels = [r'CYT', r'NUC', r'MIT', r'ME3', r'ME2', r'ME1', r'EXC', r'VAC', r'POX', r'ERL']
    
    df = pd.read_csv('datasets/yeast.csv', header=None)
    df = df.replace({'CYT':0, 'NUC':1, 'MIT':2, 'ME3':3, 'ME2':4, 'ME1':5, 'EXC':6, 'VAC':7, 'POX':8, 'ERL':9})
    data = df.iloc[:,1:9]
    target = df.iloc[:,9]
    
    sample_train, label_train, training_input, test_input = prepare_data(data, target, n, training_size, test_size, class_labels)
    
    if PLOT_DATA:
        for k in range(0,10):
            if k == 0:
                label = 'CYT'
            elif k == 1:
                label = 'NUC'
            elif k == 2:
                label = 'MIT'
            elif k == 3:
                label = 'ME3'
            elif k == 4:
                label = 'ME2'
            elif k == 5:
                label = 'ME1'
            elif k == 6:
                label = 'EXC'
            elif k == 7:
                label = 'VAC'
            elif k == 8:
                label = 'POX'
            else:
                label = 'ERL'
                
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size],
                        label=label)
        
        plt.title("Yeast Dataset")
        plt.legend()
        plt.show()
    
    return sample_train, training_input, test_input, class_labels


# ======================================================
# Parkinson's
# ======================================================
def parkinson(training_size, test_size, n, PLOT_DATA):
    class_labels = [r'Healthy', r'Parkinson\'s']
    
    df = pd.read_csv('datasets/parkinsons.csv', header=None)
    data = df.iloc[:,1:23]
    target = df.iloc[:,23]
    
    sample_train, label_train, training_input, test_input = prepare_data(data, target, n, training_size, test_size, class_labels)
    
    if PLOT_DATA:
        for k in range(0,2):
            label = 'Healthy' if k is 0 else 'Parkinson\'s'
            
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size],
                        label=label)
        
        plt.title("Parkinson's Disease Dataset")
        plt.legend()
        plt.show()

    return sample_train, training_input, test_input, class_labels

        
# ======================================================
# Heart
# ======================================================
def heart(training_size, test_size, n, PLOT_DATA):
    class_labels = [r'Absent', r'Present']
    
    df = pd.read_csv('datasets/heart.csv', header=None)
    df = df.replace({1:0, 2:1})
    data = df.iloc[:,0:13].astype(float)
    target = df.iloc[:,13].astype(float)
    
    sample_train, label_train, training_input, test_input = prepare_data(data, target, n, training_size, test_size, class_labels)
    
    if PLOT_DATA:
        for k in range(0,2):
            label = 'Absent' if k is 0 else 'Present'

            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size],
                        label=label)

        plt.title("Heart Disease Dataset")
        plt.legend()
        plt.show()

    return sample_train, training_input, test_input, class_labels