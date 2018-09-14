import numpy as np
import tensorflow as tf
import knowledge_program as kp
import gen_ML_model as ml
from sklearn import svm
from sklearn.externals import joblib
from tensorflow.examples.tutorials.mnist import input_data

inputs_file_name = 'LL_trainingdata/inputs.txt'
l1_outputs_file_name = 'LL_trainingdata/layer_1_outputs.txt'
l2_outputs_file_name = 'LL_trainingdata/layer_2_outputs.txt'
outputs_file_name = 'LL_trainingdata/outputs.txt'

def load_image_set(mnist):
    X_train = np.vstack([img.reshape(-1, ) for img in mnist.train.images])
    X_train = np.vstack([img.reshape(-1, ) for img in mnist.train.images])
    np.savetxt(inputs_file_name, X_train, "%.8f", ',')


# def calculate_outputs(input_matrix, weights, biases, excited):
#     return


# def gen_logic_learner_trainset(mnist):
#     load_image_set(mnist)
#     with tf.Session() as sess:
#         kp.load_model(sess)
#         graph = tf.get_default_graph()
#         x = graph.get_tensor_by_name("x-input:0")
#         y_ = graph.get_tensor_by_name("y-input:0")
#         train_feed = {x: mnist.train.images, y_: mnist.train.labels}
#         # 获取训练好的权重和偏倚，生成输出标签
#         w1 = graph.get_tensor_by_name('layer1/weights:0')
#         b1 = graph.get_tensor_by_name('layer1/biases:0')
#         layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
#         layer1_np = sess.run(layer1, feed_dict=train_feed)
#         #print(layer1_np.shape)
#         np.savetxt(l1_outputs_file_name, layer1_np, "%.8f", ',')
#
#         w2 = graph.get_tensor_by_name('layer2/weights:0')
#         b2 = graph.get_tensor_by_name('layer2/biases:0')
#         layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
#         layer2_np = sess.run(layer2, feed_dict=train_feed)
#         #print(layer2_np.shape)
#         np.savetxt(l2_outputs_file_name, layer2_np, "%.8f", ',')
#
#         w3 = graph.get_tensor_by_name('layer3/weights:0')
#         b3 = graph.get_tensor_by_name('layer3/biases:0')
#         layer3 = tf.matmul(layer2, w3) + b3
#         layer3_np = sess.run(layer3, feed_dict=train_feed)
#         #print(layer3_np.shape)
#         np.savetxt(outputs_file_name, layer3_np, "%.8f", ',')


def train_logic_learners(mnist):
    #获取输入矩阵
    inputs_np = np.vstack([img.reshape(-1, ) for img in mnist.test.images])
    with tf.Session() as sess:
        # 载入ML模型
        kp.load_model(sess)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        y_ = graph.get_tensor_by_name("y-input:0")
        train_feed = {x: mnist.test.images, y_: mnist.test.labels}  #10000条

        # 获取layer1输出矩阵
        w1 = graph.get_tensor_by_name('layer1/weights:0')
        b1 = graph.get_tensor_by_name('layer1/biases:0')
        layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        layer1_np = sess.run(layer1, feed_dict=train_feed)

        # # 训练layer1的logic learners
        for i in range(ml.LAYER1_NODE):
            X = inputs_np
            y = layer1_np[:, i]
            clf = svm.SVR()
            clf.fit(X, y)
            joblib.dump(clf,'LL_models/layer_1/%04d.pkl' % i)

        # 获取layer2输出矩阵
        w2 = graph.get_tensor_by_name('layer2/weights:0')
        b2 = graph.get_tensor_by_name('layer2/biases:0')
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        layer2_np = sess.run(layer2, feed_dict=train_feed)

        # 训练layer2的logic learners
        for i in range(ml.LAYER2_NODE):
            X = layer1_np
            y = layer2_np[:, i]
            clf = svm.SVR()
            clf.fit(X, y)
            joblib.dump(clf, 'LL_models/layer_2/%04d.pkl' % i)

        # 获取最终输出矩阵
        w3 = graph.get_tensor_by_name('layer3/weights:0')
        b3 = graph.get_tensor_by_name('layer3/biases:0')
        output = tf.matmul(layer2, w3) + b3
        output_np = sess.run(output, feed_dict=train_feed)

        # 训练layer2的logic learners
        for i in range(ml.OUTPUT_NODE):
            X = layer2_np
            y = output_np[:, i]
            clf = svm.SVR()
            clf.fit(X, y)
            joblib.dump(clf, 'LL_models/layer_output/%04d.pkl' % i)



def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # training_data_size = mnist.test.num_examples
    # 生成每个神经元的输入和输出数据 (可能没用)
    # gen_logic_learner_trainset(mnist)

    # 训练Logic Learners模型 （直接计算得输入数据）
    train_logic_learners(mnist)




if __name__ == '__main__':
    main()
