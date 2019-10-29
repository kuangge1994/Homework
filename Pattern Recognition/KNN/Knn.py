from scipy.io import loadmat
from collections import deque
import numpy as np

DATA_PATH = './Data/PRHW#3Data.mat'

# Traversing function for c1/c2/c3
def Traverse(i, start, stop, label, c, t1, t2, t3, queue1, queue2, queue3):
    for m in range(start, stop):
            distance1 = np.linalg.norm(c[m][:2]-t1[i][:2], ord=1)
            distance2 = np.linalg.norm(c[m][:2]-t2[i][:2], ord=1)
            distance3 = np.linalg.norm(c[m][:2]-t3[i][:2], ord=1)
            if distance1 < queue1[0][1]:
                queue1.popleft()
                queue1.append([label, distance1])
                queue1 = deque(sorted(queue1, key=lambda q:q[1], reverse=True))
            
            if distance2 < queue2[0][1]:
                queue2.popleft()
                queue2.append([label, distance2])
                queue2 = deque(sorted(queue2, key=lambda q:q[1], reverse=True))

            if distance3 < queue3[0][1]:
                queue3.popleft()
                queue3.append([label, distance3])
                queue3 = deque(sorted(queue3, key=lambda q:q[1], reverse=True))

    return queue1, queue2, queue3


# Define K-NearestNeighbor Classifier
def KNN_Class(k, c1, c2, c3, t1, t2, t3):
    # Traversing t1/t2/t3, obtain kNN results
    row = t1.shape[0]    #the numbers of samples t1/t2/t3

    for i in range(row):
        # define queues for points of top-k distance
        #respectively for t1, t2, t3; save as deque([[label, distance]])
        k_queue1 = deque([])
        k_queue2 = deque([])
        k_queue3 = deque([])
        # initial k_queue with top-k of c1, save as [label, distance]
        for m in range(k):
            distance1 = np.linalg.norm(c1[m][:2]-t1[i][:2], ord=1)
            distance2 = np.linalg.norm(c1[m][:2]-t2[i][:2], ord=1)
            distance3 = np.linalg.norm(c1[m][:2]-t3[i][:2], ord=1)
            k_queue1.append([1, distance1])
            k_queue2.append([1, distance2])
            k_queue3.append([1, distance3])
        # sorted k_queue as desending
        k_queue1 = deque(sorted(k_queue1, key=lambda q:q[1], reverse=True))
        k_queue2 = deque(sorted(k_queue2, key=lambda q:q[1], reverse=True))
        k_queue3 = deque(sorted(k_queue3, key=lambda q:q[1], reverse=True))

        # Traversing on training dataset c1, starting from c1[k]
        k_queue1, k_queue2, k_queue3 = Traverse(i,k,row,1,c1,t1,t2,t3,k_queue1,k_queue2,k_queue3)

        # Traversing on training dataset c2, starting from c2[0]
        k_queue1, k_queue2, k_queue3 = Traverse(i,0,row,2,c2,t1,t2,t3,k_queue1,k_queue2,k_queue3)

        # Traversing on training dataset c3, starting from c3[0]
        k_queue1, k_queue2, k_queue3 = Traverse(i,0,row,3,c3,t1,t2,t3,k_queue1,k_queue2,k_queue3)

        # transform deque to array for better statistics
        k_array1 = np.array(k_queue1, dtype=np.int64)
        k_array2 = np.array(k_queue2, dtype=np.int64)
        k_array3 = np.array(k_queue3, dtype=np.int64)

        # return the prelabel with the max rate
        prelabel1 = np.argmax(np.bincount(k_array1[:,0]))
        prelabel2 = np.argmax(np.bincount(k_array2[:,0]))
        prelabel3 = np.argmax(np.bincount(k_array3[:,0]))

        # apppend prelabel to t1/t2/t3; (t[i][3])
        t1[i][3] = prelabel1
        t2[i][3] = prelabel2
        t3[i][3] = prelabel3
    
    # compute the error rate of t1/t2/t3
    num_correct1 = np.sum(t1[:,3]==1)
    num_correct2 = np.sum(t2[:,3]==2)
    num_correct3 = np.sum(t3[:,3]==3)
    err_rate = 1 - (num_correct1+num_correct2+num_correct3)/(3*row)

    return err_rate


def main():
    # load data and assign them respectively
    data = loadmat(DATA_PATH)
    c1 = data['c1']
    c2 = data['c2']
    c3 = data['c3']
    t1 = data['t1']
    t2 = data['t2']
    t3 = data['t3']

    # data preprocessing; append label for c and append (label,pre_label) for t
    # label = (1, 2, 3); prrelabel = 0
    col_label = c1.shape[1]
    col_prelabel = col_label + 1
    row = c1.shape[0]
    c1 = np.insert(c1, col_label, values = np.ones(row, dtype=np.int32), axis=1)        #append label 1 for c1
    c2 = np.insert(c2, col_label, values = np.ones(row, dtype=np.int32)+1, axis=1)      #append label 2 for c2
    c3 = np.insert(c3, col_label, values = np.ones(row, dtype=np.int32)+2, axis=1)      #append label 3 for c3
    t1 = np.insert(t1, col_label, values = np.ones(row, dtype=np.int32), axis=1)        #append label 1 for t1
    t1 = np.insert(t1, col_prelabel, values = np.zeros(row, dtype=np.int32), axis=1)    #initial prelabel 0 for t1; (the same as follow)
    t2 = np.insert(t2, col_label, values = np.ones(row, dtype=np.int32)+1, axis=1)
    t2 = np.insert(t2, col_prelabel, values = np.zeros(row, dtype=np.int32), axis=1)
    t3 = np.insert(t3, col_label, values = np.ones(row, dtype=np.int32)+2, axis=1)
    t3 = np.insert(t3, col_prelabel, values = np.zeros(row, dtype=np.int32), axis=1)

    er_rate = []    #restore error rate for k=1,2,3，……,50

    # apply kNN(k-NearestNeighbor) classifier for t1, t2, t3; return error rate
    for k in range(1, 51):
        err_rate = KNN_Class(k, c1, c2, c3, t1, t2, t3)
        er_rate.append(err_rate)
        print('The error rate for {0} is {1}'.format(k, err_rate))
    
    print('----------------------------------------')
    index_min = er_rate.index(min(er_rate))
    print('The best k is {0} with error rate: {1}'.format(index_min+1, er_rate[index_min]))


if __name__ == '__main__':
    main()