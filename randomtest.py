# encoding = UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
       outputs = Wx_plus_b
    else:
       outputs = activation_function(Wx_plus_b)
    return outputs


def random(A, x, M, N):
    y_data = np.dot(A, x).reshape([1, M])
    x_data = x.reshape([1, N])
    print(x_data)

    # 2.定义节点准备接收数据
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, N])
    ys = tf.placeholder(tf.float32, [None, M])

    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
    l1 = add_layer(ys, M, N, activation_function=tf.nn.relu)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = add_layer(l1, N, N, activation_function=None)

    # 4.定义 loss 表达式
    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.square(xs - prediction))

    # 5.选择 optimizer 使 loss 达到最小
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "modelvar/csbynn1.ckpt")

        train_accuracy = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        print(train_accuracy, sess.run(prediction, feed_dict={xs: x_data, ys: y_data}))

