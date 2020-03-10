import skmultiflow
import time
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.data_stream import DataStream
from sklearn.preprocessing import MinMaxScaler

#Class for OCDD

class dataBuffer():
    def __init__(self, size, dim, percent):
        self.size = size
        self.dim = dim
        self.percent = percent
        self.win_data = np.zeros((self.size,dim))
        self.win_label = np.zeros(self.size)
        self.win_outlier= np.zeros(self.size)
        self.drift_count = 0
        self.window_index = 0
    def addInstance(self,X,y,z):
        if(self.isEmpty()):
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.win_outlier[self.window_index] = z
            self.window_index = self.window_index + 1
        else:
            self.win_data = np.roll(self.win_data, -1, axis=0)
            self.win_label = np.roll(self.win_label, -1, axis=0)
            self.win_outlier = np.roll(self.win_outlier, -1, axis=0)
            self.window_index = self.window_index - 1
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.win_outlier[self.window_index] = z   
    def driftCheck(self):
        temp, freq = np.unique(self.win_outlier, return_counts=True)
        if ((freq[0]/self.size) > self.percent): #detected
        #if (self.size-sum(int(self.win_outlier)))/self.size > percent:
            self.window_index = int(self.size * (1-self.percent))
            self.drift_count = self.drift_count + 1
            return True
        else:
            return False
    def isEmpty(self):
        return self.window_index < self.size
    def getCurrentData(self):
        return self.win_data[self.window_index:self.size]
    def getCurrentLabels(self):
        return self.win_label[self.window_index:self.size]

# Method that iterates through the dataset with given parameters
def unsupervised_analysis(df, nu, size, percent):
    stream = DataStream(df)
    stream.prepare_for_use()
    stream_clf = HoeffdingTree()
    stream_acc = []
    stream_record = []
    stream_true= 0
    buffer = dataBuffer(size, stream.n_features, percent)
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma='auto')
    
    #
    start = time.time()
    X,y = stream.next_sample(size)
    stream_clf.partial_fit(X,y, classes=stream.target_values)
    clf.fit(X)
    
    i=0
    while(stream.has_more_samples()): #stream.has_more_samples()
        X,y = stream.next_sample()
        if buffer.isEmpty():
            buffer.addInstance(X,y,clf.predict(X))
            y_hat = stream_clf.predict(X)
            stream_true = stream_true + check_true(y, y_hat)
            stream_clf.partial_fit(X,y)
            stream_acc.append(stream_true / (i+1))
            stream_record.append(check_true(y,y_hat))
            
        else:
            if buffer.driftCheck():             #detected
                #print("concept drift detected at {}".format(i))
                #retrain the model
                stream_clf.reset()
                #stream_clf = HoeffdingTree()
                stream_clf.partial_fit(buffer.getCurrentData(), buffer.getCurrentLabels(), classes=stream.target_values)
                #update one-class SVM
                clf.fit(buffer.getCurrentData())
                #evaluate and update the model
                y_hat = stream_clf.predict(X)
                stream_true = stream_true + check_true(y, y_hat)
                stream_clf.partial_fit(X,y)
                stream_acc.append(stream_true / (i+1))
                stream_record.append(check_true(y,y_hat))
                #add new sample to the window
                buffer.addInstance(X,y,clf.predict(X))
            else:
                #evaluate and update the model
                y_hat = stream_clf.predict(X)
                stream_true = stream_true + check_true(y, y_hat)
                stream_clf.partial_fit(X,y)
                stream_acc.append(stream_true / (i+1))
                stream_record.append(check_true(y,y_hat))
                #add new sample to the window
                buffer.addInstance(X,y,clf.predict(X))    
        i = i + 1
    #print(buffer.drift_count)
    
    elapsed = format(time.time() - start, '.4f')
    acc = format(stream_acc[-1] * 100, '.4f')
    final_accuracy = "Parameters: {}, {}, {}, Final accuracy: {}, Elapsed time: {}".format(nu,size,percent,acc,elapsed)
    return final_accuracy, stream_record

# Method to ignore warnings during the whole process.
def warn(*args, **kwargs):
    pass


# Making dataset ready for the process
def select_data(x):
    df = pd.read_csv(x)
    scaler = MinMaxScaler()
    df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
    return df

# Method for validating predictions of the classifier
def check_true(y,y_hat):
    if(y==y_hat):
        return 1
    else:
        return 0

def window_average(x,N):
    low_index = 0
    high_index = low_index + N
    w_avg = []
    while(high_index<len(x)):
        temp = sum(x[low_index:high_index])/N
        w_avg.append(temp)
        low_index = low_index + N
        high_index = high_index + N
    return w_avg


# MAIN CODE
warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)

df = select_data(sys.argv[1])
nu = float(sys.argv[2])
size = int(sys.argv[3])
percent = float(sys.argv[4])
stream = DataStream(df)
final_acc, st_rec = unsupervised_analysis(df,nu,size,percent)
print(final_acc)


# PLOT CODE
temp=int((len(st_rec))/30)
st_rec2 = window_average(st_rec, temp)
x = np.linspace(0, 100, len(st_rec2), endpoint=True)

f = plt.figure()
plt.plot(x, st_rec2, 'r', label='OCDD', marker="*")
plt.xlabel('Percentage of data', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True)
plt.legend(loc='lower left')
plt.ticklabel_format(style='sci')

plt.show()



