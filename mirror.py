import os
import tensorflow as tf
import numpy as np
import gen_ML_model as ml
from sklearn.externals import joblib
from tensorflow.examples.tutorials.mnist import input_data

def LL_predict(rootdir, input_np):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    output_np = np.zeros(shape=(input_np.shape[0],len(list)),dtype=float)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            print(path)
            clf = joblib.load(path)
            output_np[:, i] = clf.predict(input_np)
    return output_np

def generate_intermediate_results(inputs_np):
    '''
    生成mirror程序的中间结果和预测结果，保存在mirror_outputs文件夹中
    '''
    layer1_np = LL_predict('LL_models/layer_1', inputs_np)
    np.savetxt('mirror_outputs/layer_1.txt', layer1_np,  "%.8f", ',')
    #layer1_np = np.loadtxt('mirror_outputs/layer_1.txt',delimiter=',')

    layer2_np = LL_predict('LL_models/layer_2', layer1_np)
    np.savetxt('mirror_outputs/layer_2.txt', layer2_np, "%.8f", ',')

    y = LL_predict('LL_models/layer_output', layer2_np)
    np.savetxt('mirror_outputs/outputs.txt', y, "%.8f", ',')
    return y

def mirror_predict(mnist):
    # 获取输入矩阵
    inputs_np = np.vstack([img.reshape(-1, ) for img in mnist.validation.images])
    y = generate_intermediate_results(inputs_np)

    with tf.Session() as sess:
        y_ = tf.placeholder(tf.float32, [None, ml.OUTPUT_NODE], name='y-input')
        test_feed = {y_: mnist.validation.labels}  # 5000条
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_score = sess.run(accuracy, feed_dict=test_feed)
        print("mirror test accuracy: " + str(accuracy_score))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    mirror_predict(mnist)


if __name__ == '__main__':
    main()