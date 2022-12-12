# Created by Moontasir Abtahee at 12/12/2022

# Feature: #Enter feature name here
# Enter feature description here

# This step has to predict the mean X and the covariance P of the system state at the
# time step k . The Python function kf_predict performs the prediction of these
# output ( X and P ) when giving six input:
# X : The mean state estimate of the previous step ( k −1).
# P : The state covariance of previous step ( k −1).
# A : The transition n n × matrix.
# Q : The process noise covariance matrix.
# B : The input effect matrix.
# U : The control input.

from numpy import * #dot, sum, tile, linalg , log ,pi, exp
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time


def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X,P)

# At the time step k , this update step computes the posterior mean X and covariance
# P of the system state given a new measurement Y . The Python function kf_update
# performs the update of X and P giving the predicted X and P matrices, the
# measurement vector Y , the measurement matrix H and the measurement covariance
# matrix R . The additional input will be:
# K : the Kalman Gain matrix
# IM : the Mean of predictive distribution of Y
# IS : the Covariance or predictive mean of Y
# LH : the Predictive probability (likelihood) of measurement which is
# computed using the Python function gauss_pdf.

def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
        P = exp(-E)
    return (P[0],E[0])


#time step of mobile movement
dt = 0.1
# Initialization of state matrices
X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])

Q = eye(X.shape[0])
B = eye(X.shape[0])
U = zeros((X.shape[0],1))

# Measurement matrices
Y = array([[X[0,0] + abs(random.randn(1)[0])], [X[1,0] + abs(random.randn(1)[0])]])
H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = eye(Y.shape[0])

# Number of iterations in Kalman Filter
N_iter = 50
plt.title("Kalman Filter output")
plt.xlabel("predicted mean")
plt.ylabel("predicted covariance")

start_time = time.time()
print(start_time)
# Applying the Kalman Filter
for i in arange(0, N_iter):
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
    Y = array([[X[0,0] + abs(0.1 * random.randn(1)[0])],[X[1, 0] + abs(0.1 * random.randn(1)[0])]])
    newTime = time.time()
    print(newTime)
    # plt.plot(X,newTime,color = 'green')
    plt.plot(P,X,color = 'red')
    # plt.triplot(X)
    # plt.eventplot(P)
plt.show()
