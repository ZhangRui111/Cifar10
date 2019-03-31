import numpy as np
import time
import tensorflow as tf

import vgg19
import extract_cifar10

max_step = 20000
batch_size = 100
test_step = 200
init_learning_rate = 1e-4


# 初始化单个卷积核上的参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.Session() as sess:
    tf_x = tf.placeholder('float', [None, 32, 32, 3])
    tf_y = tf.placeholder('float', [None, 10])
    images = tf.image.resize_images(tf_x, [224, 224])

    # # input for full_layers
    x_ = tf.placeholder('float', [None, 4096])
    # learning rate decay
    global_step = tf.Variable(0, trainable=False)

    vgg = vgg19.Vgg19(vgg19_npy_path=extract_cifar10.vgg19_npy_path)
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    with tf.variable_scope('full_layer1'):
        W_fc1 = weight_variable([4096, 4096])
        # 偏置值
        b_fc1 = bias_variable([4096])
        # 将卷积的产出展开
        # pool2_flat = tf.reshape(x_, [-1, 7 * 7 * 512])
        # 神经网络计算，并添加relu激活函数
        fc1 = tf.nn.relu(tf.matmul(x_, W_fc1) + b_fc1)

    # with tf.variable_scope('full_layer2'):
    #     # 全连接第二层
    #     # 权值参数
    #     W_fc2 = weight_variable([4096, 2048])
    #     # 偏置值
    #     b_fc2 = bias_variable([2048])
    #     # 神经网络计算，并添加relu激活函数
    #     fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)

    # Dropout层，可控制是否有一定几率的神经元失效，防止过拟合，训练时使用，测试时不使用
    keep_prob = tf.placeholder("float")
    # Dropout计算
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('output_layer'):
        # 输出层，使用softmax进行多分类
        W_fc2 = weight_variable([4096, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.maximum(tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2), 1e-30)

    learning_rate = tf.train.exponential_decay(init_learning_rate,
                                               global_step=global_step,
                                               decay_steps=5000, decay_rate=0.9)
    tf.summary.scalar('LR', learning_rate)
    add_global = global_step.assign_add(1)
    # 代价函数
    cross_entropy = -tf.reduce_sum(tf_y * tf.log(y_conv))
    tf.summary.scalar('loss', cross_entropy)
    # 使用Adam优化算法来调整参数
    with tf.control_dependencies([add_global]):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # 测试正确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

    sess.run(tf.global_variables_initializer())

    # tensorboard: test&train分开记录
    merge_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(extract_cifar10.log_path + '/train', sess.graph)  # 保存位置
    test_writer = tf.summary.FileWriter(extract_cifar10.log_path + '/test', sess.graph)

    # 获取cifar10数据
    cifar10_data_set = extract_cifar10.Cifar10DataSet(extract_cifar10.cifar_10_batches_path)
    test_images, test_labels = cifar10_data_set.test_data()

    # 进行训练
    start_time = time.time()
    for i in range(max_step):
        # 获取训练数据
        batch_xs, batch_ys = cifar10_data_set.next_train_batch(batch_size)

        vgg_out = sess.run(vgg.relu6, feed_dict={tf_x: batch_xs})

        train_step.run(feed_dict={x_: vgg_out, tf_y: batch_ys, keep_prob: 0.5})

        # 每迭代500个 batch，对当前训练数据进行测试，输出当前预测准确率
        if i % 100 == 0:
            # Train_accuracy:
            vgg_out = sess.run(vgg.relu6, feed_dict={tf_x: batch_xs})

            train_accuracy = accuracy.eval(feed_dict={x_: vgg_out, tf_y: batch_ys, keep_prob: 1.0})
            print("step {0}, training accuracy {1} | learning rate: {2}".format(i, train_accuracy,
                                                                                sess.run(learning_rate)))

            _, result = sess.run([train_step, merge_op], {x_: vgg_out, tf_y: batch_ys, keep_prob: 1.0})
            train_writer.add_summary(result, i)

            # 计算间隔时间
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time

        if (i + 1) % 1000 == 0:
            # Test_accuracy
            avg = 0
            for j in range(20):
                vgg_out_test = sess.run(vgg.relu6,
                                        feed_dict={tf_x: test_images[j * batch_size:j * batch_size + batch_size]})
                avg += accuracy.eval(
                    feed_dict={x_: vgg_out_test,
                               tf_y: test_labels[j * batch_size:j * batch_size + batch_size],
                               keep_prob: 1.0})
            avg /= 20
            print("test accuracy %g" % avg)

            vgg_out_test = sess.run(
                vgg.relu6, feed_dict={tf_x: test_images[j * batch_size:j * batch_size + batch_size]})
            _, result = sess.run(
                [train_step, merge_op],
                {x_: vgg_out_test, tf_y: test_labels[j * batch_size:j * batch_size + batch_size], keep_prob: 1.0})
            test_writer.add_summary(result, i)

    # 输出整体测试数据的情况
    avg = 0
    for i in range(test_step):
        vgg_out_test = sess.run(vgg.relu6, feed_dict={tf_x: test_images[i * batch_size:i * batch_size + batch_size]})
        avg += accuracy.eval(
            feed_dict={x_: vgg_out_test,
                       tf_y: test_labels[i * batch_size:i * batch_size + batch_size],
                       keep_prob: 1.0})
    avg /= test_step
    print("test accuracy %g" % avg)
