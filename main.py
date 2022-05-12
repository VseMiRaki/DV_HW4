import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt




def p_i(X : np.ndarray, i : int, sigma=1.):
    p = np.exp(-np.linalg.norm(X[i] - X[np.concatenate((np.r_[0:i], np.r_[i+1:X.shape[0]]))], axis=-1) / 2 / (sigma**2))
    p_sum = np.sum(p)
    return p / p_sum


def getH(X : np.array, i, sigma=1.):
    P = p_i(X, i, sigma)
    H = -np.sum(P * np.log2(P + 1e-16))
    return H


def getP(X : np.ndarray, perplexity=30.0):
    logU = np.log2(perplexity)

    P_out = np.zeros((X.shape[0], X.shape[0]))
    sigma = np.ones(X.shape[0])

    for i in range(X.shape[0]):
        s_min = 0
        s_max = np.inf

        H = getH(X, i, sigma[i])
        for t in range(50):
            diff = H - logU
            if np.abs(diff) < 1e-6:
                break
            if diff > 0:
                s_max = sigma[i]
                if s_min == -np.inf:
                    sigma[i] = sigma[i] / 2.
                else:
                    sigma[i] = (sigma[i] + s_min) / 2.
            else:
                s_min = sigma[i]
                if s_max == np.inf:
                    sigma[i] = sigma[i] * 2.
                else:
                    sigma[i] = (sigma[i] + s_max) / 2.
            H = getH(X, i, sigma[i])
            if t == 49:
                print("exited because of iteration")

        
        P_out[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.shape[0]]))] = p_i(X, i, sigma[i])
    return P_out
        

def ldAffinities(Y : np.ndarray, i):
    Q = 1 / (np.linalg.norm(Y[i] - Y[np.concatenate((np.r_[0:i], np.r_[i+1:Y.shape[0]]))], axis=-1) + 1)
    sumQ = np.sum(Q)
    return Q / sumQ

def getQ(Y : np.ndarray):
    Q = np.zeros((Y.shape[0], Y.shape[0]))

    for i in range(Y.shape[0]):
        Q[i, np.concatenate((np.r_[0:i], np.r_[i+1:Y.shape[0]]))] = ldAffinities(Y, i)
    
    return Q

def getPartGrad(PQ : np.ndarray, Y : np.ndarray, i):
    delta = Y[i] - Y
    dY = (delta) / (1 + np.linalg.norm(delta))    
    return 4 * np.sum(dY * (PQ)[i][:, np.newaxis], axis=0)

def getGrad(P : np.ndarray, Q : np.ndarray, Y : np.ndarray):
    gradY = np.zeros(Y.shape)

    for i in range(Y.shape[0]):
        gradY[i, :] = getPartGrad(P - Q, Y, i)

    return gradY


def tSNE(X : np.ndarray, dim=2, perplexity=15.0):
    deltaY = 0
    eta = 1000
    alpha = 0.7
    dY = np.zeros((X.shape[0], dim))
    deltaY = np.zeros((X.shape[0], dim))
    
    P = getP(X, perplexity)
    P = (P + P.T) / X.shape[0] * 2
    P = np.maximum(P, 1e-12)

    Y = multivariate_normal.rvs(np.zeros(dim), np.identity(dim) * 1e-4, X.shape[0])

    for t in range(1000):
        Q = getQ(Y)
        Q /= X.shape[0]
        Q = np.maximum(Q, 1e-12)

        dY = getGrad(P, Q, Y)

        deltaY = alpha * deltaY - eta * dY
        Y = Y + deltaY

        if (t + 1) % 10 == 0:
            print("Iteration %d: error is %f" % (t + 1, np.sum(P * np.log(P / Q))))

        if t == 100:
            P = P / 2.

        if t == 200:
            P = P / 2.
            eta = 800

    return Y


data_path = 'data.csv'
dataset = pd.read_csv(data_path)

data = dataset[dataset.columns.difference(['Id', 'Diagnosis'])]
labels = dataset['Diagnosis']
labels.replace(['B', 'M'], [0, 1], inplace=True)

Y = tSNE(data.to_numpy())

scatter_x = Y[:, 0]
scatter_y = Y[:, 1]
group = np.array(labels.to_numpy())
cdict = {1: 'red', 0: 'blue'}
fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.savefig('result.jpg')
plt.show()





