import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_variables():
    reader = tf.train.NewCheckpointReader("model/model.ckpt")
    global_variables = reader.get_variable_to_shape_map()
    for var_name in global_variables:
        print(var_name, global_variables[var_name])#, reader.get_tensor(var_name))


def load_model(sess):
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    saver.restore(sess, 'model/model.ckpt')


def inference(graph, input_tensor):
    w1 = graph.get_tensor_by_name('layer1/weights:0')
    w2 = graph.get_tensor_by_name('layer2/weights:0')
    w3 = graph.get_tensor_by_name('layer3/weights:0')
    b1 = graph.get_tensor_by_name('layer1/biases:0')
    b2 = graph.get_tensor_by_name('layer2/biases:0')
    b3 = graph.get_tensor_by_name('layer3/biases:0')
    layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    y = tf.matmul(layer2, w3) + b3
    return y

def eval_trainset(mnist):
    with tf.Session() as sess:
        load_model(sess)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        y_ = graph.get_tensor_by_name("y-input:0")
        train_feed = {x: mnist.test.images, y_: mnist.test.labels}  # 10000条
        y = inference(graph, x)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_score = sess.run(accuracy, feed_dict=train_feed)
        print("train accuracy: " + str(accuracy_score))


def eval_testset(mnist):
    with tf.Session() as sess:
        load_model(sess)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        y_ = graph.get_tensor_by_name("y-input:0")
        test_feed = {x: mnist.validation.images, y_: mnist.validation.labels} # 5000条
        y = inference(graph, x)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_score = sess.run(accuracy, feed_dict=test_feed)
        print("test accuracy: " + str(accuracy_score))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    eval_trainset(mnist)
    eval_testset(mnist)


if __name__ == '__main__':
    main()
