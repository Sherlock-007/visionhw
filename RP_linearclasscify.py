import numpy as np
from numpy import linalg as la
from sklearn import metrics
from sklearn.svm import LinearSVC
# import cv2
from sklearn import svm
from numpy import *
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

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

clf = LinearSVC()
dataset=vstack((data1['data'],data2['data']));
dataset=vstack((dataset,data3['data']))
dataset=vstack((dataset,data4['data']))
dataset=vstack((dataset,data5['data']))

l1=data1['labels'];
l2=data2['labels'];
l3=data3['labels'];
l4=data4['labels'];
l5=data5['labels'];

l1.extend(l2);
l1.extend(l3);
l1.extend(l4);
l1.extend(l5);

dataset=dataset.T;
traindata=dot(rr,dataset);
tt=traindata[:,0:50000];
ll=l1[0:50000];
tt=tt.T

newput=data6['data'];
# newput=newput[:,0:10000];
newput=newput.T;
newput=dot(rr,newput);
newput=newput.T;

X=tt
Y=ll
lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
x_new=newput
answer=lin_clf.predict(x_new)

cnt=0
for i in range(0,x_new.shape[0]):
 if answer[i]==data6['labels'][i]:
  cnt=cnt+1

print cnt / 10000.0
# dec = lin_clf.decision_function([[1]])
# dec.shape[1]

