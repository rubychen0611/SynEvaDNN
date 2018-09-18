import os

import numpy as np
import tensorflow as tf
import knowledge_program as kp
import gen_ML_model as ml
from sklearn import svm
from sklearn.externals import joblib
from tensorflow.examples.tutorials.mnist import input_data

# logic learners 使用的回归器
LL_Regressor = svm.SVR
# logic learners 存储路径
LL_models_dir_name = 'LL_models/'

# def load_image_set(mnist):
#     X_train = np.vstack([img.reshape(-1, ) for img in mnist.train.images])
#     np.savetxt(inputs_file_name, X_train, "%.8f", ',')


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

def create_dir(dir_name):
    '''
    创建目录（若不存在）
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def train_logic_learners(trainset):
    # 创建文件夹存储中间结果
    inter_res_path = os.path.join(LL_models_dir_name, 'intermediate_results')
    create_dir(inter_res_path)
    # 获取输入矩阵
    #inputs_np = np.vstack([img.reshape(-1, ) for img in trainset.images])
    #np.savetxt(os.path.join(inter_res_path, 'inputs'), inputs_np, "%.8f", ',')
    with tf.Session() as sess:
        # 载入DNN模型
        kp.load_model(sess)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        y_ = graph.get_tensor_by_name("y-input:0")
        train_feed = {x: trainset.images, y_: trainset.labels}  #10000条
        last_layer = x
        last_layer_np = sess.run(x, feed_dict=train_feed)
        np.savetxt(os.path.join(inter_res_path, 'inputs'), last_layer_np, "%.8f", ',')
        for i in range(1, ml.HIDDEN_LAYER_NUM + 1):
            # 获取第i个隐藏层的输出矩阵
            layer_str = 'hidden_layer_%d' % i
            w = graph.get_tensor_by_name(layer_str + '/weights:0')
            b = graph.get_tensor_by_name(layer_str + '/biases:0')
            cur_layer = tf.nn.relu(tf.matmul(last_layer, w) + b)
            cur_layer_np = sess.run(cur_layer, feed_dict=train_feed)
            np.savetxt(os.path.join(inter_res_path, 'hidden_layer_'+str(i)+'_outputs'), cur_layer_np, "%.8f", ',')

            # 训练第i个隐藏层的logic learners
            ll_savepath = os.path.join(LL_models_dir_name, 'hidden_layer_' + str(i))
            create_dir(ll_savepath)
            for j in range(ml.HIDDEN_LAYER_NODE[i-1]):
                X = last_layer_np
                y = cur_layer_np[:, j]
                regressor = LL_Regressor()
                regressor.fit(X, y)
                joblib.dump(regressor , os.path.join(ll_savepath, '%04d.pkl' % j))
            last_layer = cur_layer
            last_layer_np = cur_layer_np

        # 获取最终输出矩阵
        w = graph.get_tensor_by_name('output_layer/weights:0')
        b = graph.get_tensor_by_name('output_layer/biases:0')
        output = tf.matmul(last_layer, w) + b
        output_np = sess.run(output, feed_dict=train_feed)
        np.savetxt(os.path.join(inter_res_path, 'outputs'), output_np, "%.8f", ',')

        # 训练输出层的logic learners
        savepath = os.path.join(LL_models_dir_name, 'output_layer')
        create_dir(savepath)
        for j in range(ml.OUTPUT_NODE):
            X = last_layer_np
            y = output_np[:, j]
            regressor = LL_Regressor()
            regressor.fit(X, y)
            joblib.dump(regressor, os.path.join(savepath, '%04d.pkl' % j))



def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # 生成每个神经元的输入和输出数据 (暂时没用)
    # gen_logic_learner_trainset(mnist)

    # 训练Logic Learners模型 （直接计算得输入数据）
    train_logic_learners(trainset=mnist.test)


if __name__ == '__main__':
    main()
