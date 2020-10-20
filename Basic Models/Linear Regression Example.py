import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

learning_rate = 0.01
training_step = 1000
display_step = 50

X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

W = tf.Variable(rng.randn(), name='weight')
b = tf.Variable(rng.randn(), name='bias')


def linear_regression(x):
    return W * x + b


def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Stochastic Gradient Descent Optimizer.
optimizer = tf.optimizers.SGD(learning_rate)  # 随机梯度下降


# 使用梯度带GradientTape()自动求导
def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # 计算梯度
    gradients = g.gradient(loss, [W, b])
    # 更新梯度
    optimizer.apply_gradients(zip(gradients, [W, b]))


for step in range(1, training_step + 1):
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step:%i,loss:%f,W:%f,b:%f" % (step, loss, W.numpy(), b.numpy()))

plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()
