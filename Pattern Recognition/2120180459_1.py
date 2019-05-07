import numpy as np 
import matplotlib.pyplot as plt 
import math 
import random

# The number of samples
Number = 1000

# The mean of three Distributions
mean1 = [1, 1]
mean2 = [4, 4]
mean3 = [8, 1]

# The convariation of three Distribution models
Cov = [[2, 0], [0, 2]]

# The risk factor, decided i while is j
risk = [[0, 2, 3], [1, 0, 2.5], [1, 1, 0]]

# Prior probability of two sets X and X'
Pw1 = [1/3, 1/3, 1/3]
Pw2 = [0.6, 0.3, 0.1]

# Count number of three Distributions
def Count(Pw = [0, 0, 0]):
    count_1 = count_2 = count_3 = 0
    
    for i in range(Number):
        p = random.uniform(0, 1)
        if p < (Pw[0]):
            count_1 += 1
            continue
        elif p < (Pw[0] + Pw[1]):
            count_2 += 1
            continue
        else:
            count_3 += 1
            continue
    return count_1, count_2, count_3

# Plot the initial distribution
def Draw(M_1=[], M_2=[], M_3=[]):
    plt.plot(M_1[:,0], M_1[:,1], '.', c='red', label='m1: '+str(len(M_1)))
    plt.plot(M_2[:,0], M_2[:,1], '.', c='green', label='m2: '+str(len(M_2)))
    plt.plot(M_3[:,0], M_3[:,1], '.', c='blue', label='m3: '+str(len(M_3)))
    plt.legend()
    

# Generate three distribution for sets(X or X')
# set_num = 1 --> X; set_num = 2 --> X' for Draw(M_1, M_2, M_3)
def Generate(Pw = [0 , 0, 0], set_num = 0):
    count1 = count2 = count3 = 0
    count1, count2, count3 =Count(Pw)

    # Generate data, combine them to X
    M_1 = np.random.multivariate_normal(mean1, Cov, count1)
    M_2 = np.random.multivariate_normal(mean2, Cov, count2)
    M_3 = np.random.multivariate_normal(mean3, Cov, count3)
    X = np.vstack((M_1, M_2, M_3))

    # Draw the distribution of sets
    if set_num == 1:
        plt.subplot(2,5,1)
        plt.title('DataSet X')
        Draw(M_1, M_2, M_3)
    elif set_num == 2:
        plt.subplot(2,5,6)
        plt.title('DataSet X\'')
        Draw(M_1, M_2, M_3)
    else:
        print("Error!!! set_num is not in [1, 2]")


    # Generate Labels of X
    Label = []
    for i in range(0, count1):
        Label.append(1)
    for i in range(0, count2):
        Label.append(2)
    for i in range(0, count3):
        Label.append(3)
    
    return X, Label

# Define the likelihood function
# Parameters: X[], Cov, mean
def Likehood(X, Cov, mean):
    stand_dev = Cov[0][0] * Cov[1][1]   #Cov[0][1] = Cov[1][0] = 0
    x_u = [0, 0]
    x_u[0] = X[0]-mean[0]
    x_u[1] = X[1]-mean[1]
    e_para = np.mat(x_u)*(np.mat(Cov)).I*(np.mat(x_u)).T
    fun = (1 / (2 * math.pi * math.sqrt(stand_dev))) * np.e**(-0.5 * e_para.tolist()[0][0])
    return fun

# Likelihood rate test rule (LRT)
def LRT(X=[], Pw=[0, 0, 0]):
    lr = [0, 0, 0]      #initial the result of LRT for three distributions
    pred_lab = []       #initial the label for the result of Predicted

    for i in range(0, Number):
        lr[0] = Likehood(X[i], Cov, mean1) * Pw[0]
        lr[1] = Likehood(X[i], Cov, mean2) * Pw[1]
        lr[2] = Likehood(X[i], Cov, mean3) * Pw[2]
        Max = max(lr)
        label = lr.index(Max) + 1
        pred_lab.append(label)

    return pred_lab    

# Bayesian risk test rules (BRT)
def BRT(X=[], Pw=[0, 0, 0]):
    pred_lab = []       #initial the label for the result of predicted
    prob = [0, 0, 0]    #restore P(x|w_j)P(w_j)

    for i in range(0, Number):
        br = [0, 0, 0] # initial the result of BRT for three distributions
        prob[0] = Likehood(X[i], Cov, mean1) * Pw[0]
        prob[1] = Likehood(X[i], Cov, mean2) * Pw[1]
        prob[2] = Likehood(X[i], Cov, mean3) * Pw[2]
        for m in range(0, 3):
            for n in range(0, 3):
                br[m] += risk[m][n] * prob[n]
        Min = min(br)
        label = br.index(Min) + 1
        pred_lab.append(label)
    
    return pred_lab    

# Maximum posterior probability test rule (MAP)
def MAP(X=[], Pw=[0, 0, 0]):
    ma = [0, 0, 0]      #initial the result of MAP for three distributions
    pred_lab = []       #initial the label for the result of predicted
    prob = [0, 0, 0]    #restore P(x|w_j)P(w_j)

    for i in range(0, Number):
        prob[0] = Likehood(X[i], Cov, mean1) * Pw[0]
        prob[1] = Likehood(X[i], Cov, mean2) * Pw[1]
        prob[2] = Likehood(X[i], Cov, mean3) * Pw[2]
        ma[0] = prob[0] / sum(prob)
        ma[1] = prob[1] / sum(prob)
        ma[2] = prob[2] / sum(prob)
        Max = max(ma)
        label = ma.index(Max) + 1
        pred_lab.append(label)

    return pred_lab  

# Minimum Euclidean distance rule (MET)
def MET(X=[], Pw=[0, 0, 0]):
    me = [0, 0, 0]      #initial the result of MET for three distribution
    pred_lab = []       #initial the label for the result of predicted

    for i in range(0, Number):
        me[0] = (X[i][0] - mean1[0])**2 + (X[i][1] - mean1[1])**2
        me[1] = (X[i][0] - mean2[0])**2 + (X[i][1] - mean2[1])**2
        me[2] = (X[i][0] - mean3[0])**2 + (X[i][1] - mean3[1])**2
        Min = min(me)
        label = me.index(Min) + 1
        pred_lab.append(label)

    return pred_lab

# Show the result of predicted
def Pred_Show(X=[], Label=[], Pred_lab=[]):
    error = 0       #count the points of error classified

    for i in range(0, Number):
        if Pred_lab[i] == Label[i]:
            if Label[i] == 1:
                plt.plot(X[i][0], X[i][1], '.', c='red')
            elif Label[i] == 2:
                plt.plot(X[i][0], X[i][1], '.', c='green')
            else:
                plt.plot(X[i][0], X[i][1], '.', c='blue')
        elif Pred_lab[i] != Label[i]:
            error += 1
            if Label[i] == 1:
                plt.plot(X[i][0], X[i][1], '*', c='black')
            elif Label[i] == 2:
                plt.plot(X[i][0], X[i][1], '*', c='orange')
            else:
                plt.plot(X[i][0], X[i][1], '*', c='purple')
    plt.legend()

    return error / Number

def main():
    # Define a canvas, size of 50*20, resolution of 50
    plt.figure(num=1, figsize=(50,20), dpi=50)
    
    # Create the distribution of X, return X(as X_1) and Label_1
    X_1, Label_1 = Generate(Pw1, 1)
    print("--------------------DataSet X--------------------")

    # Compute Predicted label for LRT, Set X
    Pred_LRT = []
    Pred_LRT = LRT(X_1, Pw1)
    plt.subplot(2,5,2)
    plt.title('LRT for X')
    err_rate = Pred_Show(X_1, Label_1, Pred_LRT)
    print("the LRT error rate of X is: %.3f" % err_rate)

    # Compute Predicted label for BRT, Set X
    Pred_BRT = []
    Pred_BRT = BRT(X_1, Pw1)
    plt.subplot(2,5,3)
    plt.title('BRT for X')
    err_rate = Pred_Show(X_1, Label_1, Pred_BRT)
    print("the BRT error rate of X is: %.3f" % err_rate)

    # Compute Predicted label for MAP, Set X
    Pred_MAP = []
    Pred_MAP = MAP(X_1, Pw1)
    plt.subplot(2,5,4)
    plt.title('MAP for X')
    err_rate = Pred_Show(X_1, Label_1, Pred_MAP)
    print("the MAP error rate of X is: %.3f" % err_rate)

    # Compute Predicted label for METT, Set X
    Pred_MET = []
    Pred_MET = MET(X_1, Pw1)
    plt.subplot(2,5,5)
    plt.title('MET for X')
    err_rate = Pred_Show(X_1, Label_1, Pred_MET)
    print("the MET error rate of X is: %.3f" % err_rate)


    # Create the distribution of X', return X'(as X_2) and Label_2
    X_2, Label_2 = Generate(Pw2, 2)
    print("--------------------DataSet X'--------------------")

    # Compute Predicted label for LRT, Set X'
    Pred_LRT = []
    Pred_LRT = LRT(X_2, Pw2)
    plt.subplot(2,5,7)
    plt.title('LRT for X\'')
    err_rate = Pred_Show(X_2, Label_2, Pred_LRT)
    print("the LRT error rate of X' is: %.3f" % err_rate)

    # Compute Predicted label for BRT, Set X'
    Pred_BRT = []
    Pred_BRT = BRT(X_2, Pw2)
    plt.subplot(2,5,8)
    plt.title('BRT for X\'')
    err_rate = Pred_Show(X_2, Label_2, Pred_BRT)
    print("the BRT error rate of X' is: %.3f" % err_rate)

    # Compute Predicted label for MAP, Set X'
    Pred_MAP = []
    Pred_MAP = MAP(X_2, Pw2)
    plt.subplot(2,5,9)
    plt.title('MAP for X\'')
    err_rate = Pred_Show(X_2, Label_2, Pred_MAP)
    print("the MAP error rate of X' is: %.3f" % err_rate)

    # Compute Predicted label for MET, Set X'
    Pred_MET = []
    Pred_MET = MET(X_2, Pw2)
    plt.subplot(2,5,10)
    plt.title('MET for X\'')
    err_rate = Pred_Show(X_2, Label_2, Pred_MET)
    print("the MET error rate of X' is: %.3f" % err_rate)
    
    plt.show()


if __name__ == '__main__':
    main()