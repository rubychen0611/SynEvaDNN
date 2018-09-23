import os
import tensorflow as tf
import numpy as np
import gen_ML_model as ml
import program_synthesis as ps
from sklearn.externals import joblib
from tensorflow.examples.tutorials.mnist import input_data

# 预测中间结果保存目录
mirror_outputs_dir = 'predict/mirror_outputs/'


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
    cur_layer_np = inputs_np
    for i in range(1, ml.HIDDEN_LAYER_NUM + 1):
        modelpath = os.path.join(ps.LL_models_dir_name , 'hidden_layer_' + str(i))
        cur_layer_np = LL_predict(modelpath, cur_layer_np)
        np.savetxt(os.path.join(mirror_outputs_dir, 'hidden_layer_' + str(i)), cur_layer_np,  "%.8f", ',')

    modelpath = os.path.join(ps.LL_models_dir_name, 'output_layer')
    cur_layer_np = LL_predict(modelpath, cur_layer_np)
    np.savetxt(os.path.join(mirror_outputs_dir,'final_outputs'), cur_layer_np, "%.8f", ',')
    return cur_layer_np


def mirror_predict(testset):
    '''
    使用mirror程序在测试数据集上预测并输出结果
    '''
    # 获取输入矩阵
    inputs_np = np.vstack([img.reshape(-1, ) for img in testset.images])
    y = generate_intermediate_results(inputs_np)

    with tf.Session() as sess:
        y_ = tf.placeholder(tf.float32, [None, ml.OUTPUT_NODE], name='y-input')
        test_feed = {y_: testset.labels}  # 5000条
        accuracy = ml.calc_accuracy(y, y_)
        accuracy_score = sess.run(accuracy, feed_dict=test_feed)
        print("mirror test accuracy: " + str(accuracy_score))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    mirror_predict(testset=mnist.validation)


if __name__ == '__main__':
    main()