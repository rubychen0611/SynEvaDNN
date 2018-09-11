import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_NODE = 784  # 输入节点(28*28)
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 300  # 隐藏层1节点数
LAYER2_NODE = 100  # 隐藏层2节点数

# 模型相关的参数
# LEARNING_RATE_BASE = 0.8
# LEARNING_RATE_DECAY = 0.99
# REGULARAZTION_RATE = 0.0001
BATCH_SIZE = 20  # 每次batch打包的样本个数
TRAINING_STEPS = 5000

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"

def get_weight_variable(shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weights")
    return weights


def inference(input_tensor):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE])
        biases = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]), name="biases")
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE])
        biases = tf.Variable(tf.constant(0.0, shape=[LAYER2_NODE]), name="biases")
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    with tf.variable_scope('layer3'):
        weights = get_weight_variable([LAYER2_NODE, OUTPUT_NODE])
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]), name="biases")
        layer3 = tf.matmul(layer2, weights) + biases
    return layer3


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 计算前向传播结果
    y = inference(x)

    # 定义训练轮数
    global_step = tf.Variable(0, trainable=False, name="global_step")

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy, name='loss')

    # 设置学习率。
    learning_rate = 0.01

    # 优化损失函数
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))
        # 保存模型
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        print(sess.run(tf.get_default_graph().get_tensor_by_name('layer1/weights:0')))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
