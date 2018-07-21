# encoding = UTF-8

import numpy as np
from scipy import integrate

M = 200
N = 1000
npre = 1


def g(m, x):
    n = np.zeros(len(m))
    for j in range(len(m)):
        if m[j] >= x:
            n[j] = m[j] - x
        elif m[j] <= -x:
            n[j] = m[j] + x
        else:
            n[j] = 0

    return n


def pg(m, x):
    n = np.zeros(len(m))
    for j in range(len(m)):
        if m[j] >= x or m[j] <= -x:
            n[j] = 1
        else:
            n[j] = 0

    return np.average(n)


def fa(r, v):
    y = np.zeros(len(r))
    for i in range(len(r)):
        dis = lambda x: np.exp(- npre * abs(x) - v * (x-r[i])**2/2)
        rho = integrate.quad(dis, -np.inf, np.inf)[0]
        f = lambda x: x * (np.exp(- npre * abs(x) - v * (x-r[i])**2/2))/ rho
        y[i] = integrate.quad(f, -np.inf, np.inf)[0]

    return y


def fc(r, v):
    y = np.zeros(len(r))
    for i in range(len(r)):
        dis = lambda x: np.exp(- npre * abs(x) - v * (x - r[i]) ** 2 / 2)
        rho = integrate.quad(dis, -np.inf, np.inf)[0]
        f1 = lambda x: x**2 * (np.exp(- npre * abs(x) - v * (x - r[i]) ** 2 / 2))/rho
        f = lambda x: x * (np.exp(- npre * abs(x) - v * (x - r[i]) ** 2 / 2))/rho
        y[i] = v * (integrate.quad(f1, -np.inf, np.inf)[0] - integrate.quad(f, -np.inf, np.inf)[0]**2)

    return np.average(y)


def vamp(A, x):
    y = np.dot(A, x)

    R = np.linalg.matrix_rank(A)
    u, s, vT = np.linalg.svd(A, full_matrices=False)
    sd = np.diag(s)

    y1 = np.dot(np.dot(np.linalg.inv(sd), u.transpose()), y)

    seed = 5
    r = np.random.random_sample(N)
    gamma = 0.5

    for k in range(100):
        xk = g(r, gamma)
        alpha = pg(r, gamma)
        print("a:", alpha)
        rtilde = (xk - alpha * r) / (1.0 - alpha)
        gammak = gamma * (1.0 - alpha) / alpha
        print("gk:",gammak)
        dk = npre * np.dot(np.linalg.inv(np.diag(npre * s * s + gammak * np.ones(R))), s * s)
        gamma = gammak * R * np.average(dk) / (N - R * np.average(dk))
        print("g:", gamma)
        r = rtilde + N * np.dot(np.dot(vT.transpose(), np.diag(dk / np.average(dk))), (y1 - np.dot(vT, rtilde))) / R
        t = xk - x
        result = np.average([i*i for i in t])
        print("result:", result)
        if result <= 10**(-10):
            break

    return xk, result, k


def amp(A, x):
#    u, s, vT = np.linalg.svd(A, full_matrices=False)
    y = np.dot(A, x)
#    y = np.dot(np.dot(np.linalg.inv(np.diag(s)), u.transpose()), y)
#    A = vT
    R = np.dot(A.transpose(), y)
    V = 0.01
    a = np.zeros(N)
    z = np.zeros(M)

    for k in range(100):
        z = y - np.dot(A, a) + fc(R, V) * N * z / M
        V = fc(R, V) * N * V / M
        print(V)
        R = a + np.dot(A.transpose(), z)
        a = fa(R, V)
        t = a-x
        result = np.average([i*i for i in t])
        print(result)
        if result <= 10**(-10):
            break

    return a, result, k

