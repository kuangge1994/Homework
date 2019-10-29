import numpy as np 
import math
from matplotlib import pyplot as plt 

# The mean of three Distributions
mean1 = [0.1, 0.1]
mean2 = [2.1, 1.9]
mean3 = [-1.5, 2.0]

# The convariation of three Distribution models
Cov = [[1.2, 0.4], [0.4, 1.8]]

# Prior probability of three distributions
Pw = [1/3, 1/3, 1/3]

# Define the likelihood function
# Parameters: X[], Cov, mean
def Likelihood(X, Cov, mean):
    stand_dev = Cov[0][0]*Cov[1][1] - Cov[0][1]*Cov[1][0]
    x_u = [0, 0]
    x_u[0] = X[0] - mean[0]
    x_u[1] = X[0] - mean[1]
    e_para = np.mat(x_u) * (np.mat(Cov)).I * (np.mat(x_u)).T
    result = (1 / (2*math.pi*math.sqrt(stand_dev))) * np.e**(-0.5 * e_para.tolist()[0][0])
    return result

# Categorizer
def Classification(x, Pw):
    Pos = [0, 0, 0]
    # compute maximum posterior probability for three distribution
    Pos[0] = Likelihood(x, Cov, mean1) * Pw[0]
    Pos[1] = Likelihood(x, Cov, mean2) * Pw[1]
    Pos[2] = Likelihood(x, Cov, mean3) * Pw[2]
    Max = max(Pos)
    label = Pos.index(Max) + 1
    return label

# Function of Contour plot
def Plt_contor(u, Cov):
    y, x = np.ogrid[0:4:40j, 0:4:40j]
    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    # x_dev = x - u[0]
    # y_dev = y - u[1]
    # Ma_dst = (np.mat([x_dev, y_dev]) * (np.mat(Cov)).I * (np.mat([x_dev, y_dev])).T).tolist()[0][0]
    Ma_dst = (np.mat([x-u[0], y-u[1]]) * (np.mat(Cov)).I * (np.mat([x-u[0], y-u[1]])).T).tolist()[0][0]
    plt.figure(num=1,figsize=(15,15),dpi=60)
    plt.plot(u[0], u[1], c='red', marker='o')
    plt.text(u[0], u[1], (u[0], u[1]))
    cs = plt.contour(Ma_dst, 50, extent=extent)
    plt.clabel(cs)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Classify u=[1.6, 1.5]
# Plot the Mahalanobis distance for characteristic vector: mean2=[2.1, 1.9]
def main():
    print('---------- classify vector=[1.6, 1.5] ----------')
    # unknown vector
    u_vector = [1.6, 1.5]

    # obtain the category of u_vector
    label = Classification(u_vector, Pw)
    print('The label of [%.1f, %.1f] is: %d' %(u_vector[0], u_vector[1], label))

    # plot the contor line for u1=[2.1, 1.9]
    print('---- Plot the contour lines for [2.1, 1.9] ----')
    Plt_contor(mean2, Cov)

if __name__ == '__main__':
    main()