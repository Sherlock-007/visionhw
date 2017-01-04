import numpy as np
import pylab as pl
from StringIO import StringIO
from numpy import *
import random
import operator
import  math
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
def createDataSet():
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A','A','B','B']
    return group,labels


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row
    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in xrange(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


# group,labels = createDataSet()
# print group.shape
# print type(labels)
# t = kNNClassify([0,0],group,labels,3)
# print t

data1=unpickle('/home/sherlock/PycharmProjects/hw1/cifar-10-batches-py/data_batch_1');
data2=unpickle('/home/sherlock/PycharmProjects/hw1/cifar-10-batches-py/data_batch_2');
data3=unpickle('/home/sherlock/PycharmProjects/hw1/cifar-10-batches-py/data_batch_3');
data4=unpickle('/home/sherlock/PycharmProjects/hw1/cifar-10-batches-py/data_batch_4');
data5=unpickle('/home/sherlock/PycharmProjects/hw1/cifar-10-batches-py/data_batch_5');
data6=unpickle('/home/sherlock/PycharmProjects/hw1/cifar-10-batches-py/test_batch');
dimension=500
rr=np.zeros(dimension*3072);
rr=rr.reshape(dimension,3072);

for i in range(1,dimension):

   for j in range(1,3072):
    xx=random.uniform(0,1)
    if xx<1.0/6:
     # print '1'
     rr[i,j]=1

    elif xx<5.0/6:
     # print '0'
     rr[i,j]=0
    else :
     # print '-1'
     rr[i,j]=-1

rr=sqrt(3)*rr;
# for i in range(1,200):
#     for j in range(1,3072):
#      print rr[i,j],;
#      print ' ',
#     print '\n'
# print rr.shape

dataset=vstack((data1['data'],data2['data']));
dataset=vstack((dataset,data3['data']))
dataset=vstack((dataset,data4['data']))
dataset=vstack((dataset,data5['data']))
print dataset.dtype
l1=data1['labels'];
l2=data2['labels'];
l3=data3['labels'];
l4=data4['labels'];
l5=data5['labels'];
# label=vstack((l1,l2));
# label=vstack((label,l3));
# label=vstack((label,l4));
# label=vstack((label,l5));
# label=label.reshape(1,50000)
l1.extend(l2);
l1.extend(l3);
l1.extend(l4);
l1.extend(l5);
print l1
dataset=dataset.T;
traindata=dot(rr,dataset);
tt=traindata[:,0:50000];
ll=l1[0:50000];
tt=tt.T
print tt.shape
print tt.dtype
print type(tt)
print len(ll)
# ans=linalg.svd(dataset,0,1);
newput=data6['data'];
# newput=newput[:,0:10000];
newput=newput.T;
newput=dot(rr,newput);
newput=newput.T;
print newput.shape
# print label.shape
# print type(label)

#ss=float(newput);
#print ss.dtype;
result=zeros(10000);
cnt=0
for i in range(0,newput.shape[0]):

 result[i]=kNNClassify(newput[i,:],tt,ll,9);
 if result[i]==data6['labels'][i]:
  cnt=cnt+1



print cnt/10000.0

