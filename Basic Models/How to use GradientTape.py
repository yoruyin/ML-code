import tensorflow as tf


def gradient_test1():
    print("yes")
    x = tf.constant(3.0)
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch(x)
        y1 = 2 * x
        y2 = x * x + 2
        y3 = x * x + 2 * x

    dy1_dx = tape.gradient(target=y1, sources=x)
    dy2_dx = tape.gradient(target=y2, sources=x)
    dy3_dx = tape.gradient(target=y3, sources=x)

    print("dy1_dx:", dy1_dx)
    print("dy2_dx:", dy2_dx)
    print("dy3_dx:", dy3_dx)


def gradient_test2():
    x = tf.constant(3.0)
    y = tf.constant(2.0)
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch([x, y])
        z1 = x * x * y + x * y

    dz1_dx = tape.gradient(target=z1, sources=x)
    dz1_dy = tape.gradient(target=z1, sources=y)
    dz1_dxdy = tape.gradient(target=z1, sources=[x, y])
    print("dz1_dx:", dz1_dx)
    print("dz1_dy:", dz1_dy)
    print("dz1_dxdy:", dz1_dxdy)


if __name__ == "__main__":
    gradient_test2()
