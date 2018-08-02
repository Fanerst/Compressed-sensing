# encoding = UTF-8

import numpy as np
import message_passing
import nn
import randomtest

# setting constant variables
M = 200
N = 1000
d = 30

# initialize matrix A and x
A = np.zeros((M, N))
x = np.zeros(N)
x_sample = np.zeros([N, 10000])
y_sample = np.zeros([M, 10000])

with open('data/AAA_Gauss_1000_1000.mtx', 'r') as f1:
    list1 = f1.readlines()
with open('data/XXX0samples_Gaussian_1000.dat', 'r') as f2:
    list2 = f2.readlines()
f1.close()
f2.close()

# constructing Gaussian matrix A
for i in range(2, M+2):
    A[i-2] = list1[i].split()
for j in range(N):
    A[:, j] = A[:, j] / np.sqrt(sum([k*k for k in A[:, j]]))

for i in range(3, d+3):
    temp = list2[i].split()
    x[int(temp[0])-1] = float(temp[1])

# construct samples of the original vector x and calculate the compressed vector y
s = np.random.randint(4, 1003, [d, 10000])
for j in range(10000):
    for i in s[:, j]:
        temp = list2[i].split()
        x_sample[int(temp[0]), j] = float(temp[1])

for i in range(10000):
    y_sample[:, i] = np.dot(A, x_sample[:, i])

# generating random matrix
A1 = np.random.random_sample((M, N))
for i in range(M):
    A1[i, :] = A1[i, :] / np.sqrt(sum([k*k for k in A1[i, :]]))

# x1, result1, k1 = message_passing.amp(A, x)
x2, result2, k2 = message_passing.vamp(A1, x)

print(result2, k2)

# nn.nn(A, x, M, N)
