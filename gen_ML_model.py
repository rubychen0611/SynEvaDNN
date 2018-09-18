import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 模型相关的参数
INPUT_NODE = 784  # 输入节点数(28*28)
OUTPUT_NODE = 10  # 输出节点数
HIDDEN_LAYER_NUM = 2 # 隐藏层数目
HIDDEN_LAYER_NODE = [20, 20] # 每个隐藏层节点数

# LEARNING_RATE_BASE = 0.8
# LEARNING_RATE_DECAY = 0.99
# REGULARAZTION_RATE = 0.0001
BATCH_SIZE = 20  # 每次batch打包的样本个数
TRAINING_STEPS = 5000 # 训练轮数

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


def get_weight_variable(shape):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return weights


def get_biases_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.0))
    return biases


def inference(input_tensor):
    '''
    计算前向传播结果
    '''
    # 隐藏层layer1
    with tf.variable_scope('hidden_layer_1'):
        weights = get_weight_variable([INPUT_NODE, HIDDEN_LAYER_NODE[0]])
        biases = get_biases_variable([HIDDEN_LAYER_NODE[0]])
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 隐藏层layer2-layerN
    if HIDDEN_LAYER_NUM > 1:
        last_layer = layer1
        for i in range(1, HIDDEN_LAYER_NUM):
            with tf.variable_scope('hidden_layer_'+str(i+1)):
                weights = get_weight_variable([HIDDEN_LAYER_NODE[i-1], HIDDEN_LAYER_NODE[i]])
                biases = get_biases_variable([HIDDEN_LAYER_NODE[i]])
                cur_layer = tf.nn.relu(tf.matmul(last_layer, weights) + biases)
                last_layer = cur_layer

    # 输出层
    with tf.variable_scope('output_layer'):
        weights = get_weight_variable([HIDDEN_LAYER_NODE[HIDDEN_LAYER_NUM-1], OUTPUT_NODE])
        biases = get_biases_variable([OUTPUT_NODE])
        layer3 = tf.matmul(last_layer, weights) + biases
    return layer3


def calc_accuracy(y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def train(trainset, testset):
    '''
    训练模型
    '''

    # 输入数据和标签
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
    accuracy = calc_accuracy(y, y_)

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        #validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: testset.images, y_: testset.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            xs, ys = trainset.next_batch(BATCH_SIZE)  # 用test数据集训练(10000个)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            # if i % 1000 == 0:
            #     validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            #     print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
        test_acc = sess.run(accuracy, feed_dict=test_feed) # 用validate数据集测试(5000个)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))
        # 保存模型
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # 10000条训练数据，5000条测试数据
    train(trainset=mnist.test, testset=mnist.validation)


if __name__ == '__main__':
    main()
