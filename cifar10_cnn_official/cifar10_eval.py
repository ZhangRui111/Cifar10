# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

###
# tf.app.flags.DEFINE_xxx()就是添加命令行的optional argument（可选参数），而tf.app.flags.FLAGS
# 可以从对应的命令行参数取出参数。
# (key, value, explanation)
###
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './log/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './log/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    ###
    # * Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。
    #   Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值 。只要提供一个
    #   计数器，当计数器触发时，Saver类可以自动的生成checkpoint文件。
    # * 这让我们可以在训练过程中保存多个中间结果。例如，我们可以保存每一步训练的结果。
    # * 为了避免填满整个磁盘，Saver可以自动的管理Checkpoints文件。例如，我们可以指定
    #   保存最近的N个Checkpoints文件。
    ###
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # # 判定ckpt和ckpt.model_checkpoint_path是否都存在
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it. 即上面路径中最后的：0
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        # # 线程队列参照：http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/threading_and_queues.html
        coord = tf.train.Coordinator()
        threads = []
        try:
            # # 所有队列管理器被默认加入图的tf.GraphKeys.QUEUE_RUNNERS集合中。
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            # # math.ceil() 函数返回数字的上入整数。
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])  # # 一次计算一个batch_size的top_k_op
                true_count += np.sum(predictions)  # # np.sum()把True看做1，False看做0.
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # # inference model. 推理模型
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        ###
        # tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，
        # tf.nn.in_top_k(prediction, target, K):
        # prediction就是表示你预测的结果，大小就是预测样本的数量乘以输出的维度，类型是tf.float32
        # 等。target就是实际样本类别的标签，大小就是样本数量的个数。K表示每个样本的预测结果的前K
        # 个最大的数里面是否含有target中的值。一般都是取1。
        #
        # import tensorflow as tf;
        # A = [[0.8,0.6,0.3], [0.1,0.6,0.4]]
        # B = [1, 1]
        # out = tf.nn.in_top_k(A, B, 1)
        # with tf.Session() as sess:
        #     sess.run(tf.initialize_all_variables())
        #     print sess.run(out)
        #
        # Running Result:[False, True] ! 注意0起址index
        ###

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        ###
        # tf.train.ExponentialMovingAverage这个函数用于更新参数，就是采用滑动平均的方法更新参数。
        # 这个函数初始化需要提供一个衰减速率（decay），用于控制模型的更新速度。这个函数还会维护一个
        # 影子变量（也就是更新参数后的参数值），这个影子变量的初始值就是这个变量的初始值，影子变量
        # 值的更新方式如下：
        # shadow_variable = decay * shadow_variable + (1-decay) * variable
        # shadow_variable是影子变量，variable表示待更新的变量，也就是变量被赋予的值，decay为衰减速率。
        # decay一般设为接近于1的数（0.99,0.999）。decay越大模型越稳定，因为decay越大，参数更新的速度
        # 就越慢，趋于稳定。
        ###
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            # # 休眠一顿时间后再评估
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()  # Download and extract the tarball from Alex's website.
    # 若已存在eval_dir指示的文件夹，就删除再新建
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
