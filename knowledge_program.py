import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import gen_ML_model as ml

def get_variables():
    '''
   输出所有变量名和变量值，调试用
    '''
    reader = tf.train.NewCheckpointReader("model/model.ckpt")
    global_variables = reader.get_variable_to_shape_map()
    for var_name in global_variables:
        print(var_name, global_variables[var_name], reader.get_tensor(var_name))


def load_model(sess):
    '''
    载入保存的机器学习模型
    '''
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    saver.restore(sess, 'model/model.ckpt')


def inference(graph, input_tensor):
    '''
    计算前向传播结果
    '''
    layer = input_tensor
    for i in range(1, ml.HIDDEN_LAYER_NUM + 1):
        layer_str = 'hidden_layer_%d' % i
        w = graph.get_tensor_by_name(layer_str + '/weights:0')
        b = graph.get_tensor_by_name(layer_str + '/biases:0')
        layer = tf.nn.relu(tf.matmul(layer, w) + b)

    w = graph.get_tensor_by_name('output_layer/weights:0')
    b = graph.get_tensor_by_name('output_layer/biases:0')
    y = tf.matmul(layer, w) + b
    return y


def eval_accuracy(testset):
    '''
   计算kp在数据集上的准确率
    '''
    with tf.Session() as sess:
        load_model(sess)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        y_ = graph.get_tensor_by_name("y-input:0")
        test_feed = {x: testset.images, y_: testset.labels}
        y = inference(graph, x)
        accuracy = ml.calc_accuracy(y, y_)
        accuracy_score = sess.run(accuracy, feed_dict=test_feed)
        return accuracy_score


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # 输出训练集预测准确率
    train_accuracy = eval_accuracy(mnist.test)
    print("train accuracy: " + str(train_accuracy))
    # 输出测试集预测准确率
    test_accuracy = eval_accuracy(mnist.validation)
    print("test accuracy: " + str(test_accuracy))


if __name__ == '__main__':
    main()
