import tensorflow as tf
import gen_model as gm
from tensorflow.examples.tutorials.mnist import input_data

def get_variables():
    #saver = tf.train.import_meta_graph("model/model.ckpt.meta")
    reader = tf.train.NewCheckpointReader("model/model.ckpt")
    global_variables = reader.get_variable_to_shape_map()
    for var_name in global_variables:
        print(var_name, global_variables[var_name], reader.get_tensor(var_name))


def load_model():
    pass

def predict_one():
    pass


def eval_trainset(mnist):
    pass


def eval_testset(mnist):
    x = tf.placeholder(tf.float32, [None, gm.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, gm.OUTPUT_NODE], name='y-input')
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    y = gm.inference(x)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/model.ckpt.meta')
        saver.restore(sess, 'model/model.ckpt')
        graph = tf.get_default_graph()
        w = graph.get_tensor_by_name('layer2/biases:0')
        print(sess.run(w))
        #accuracy_score = sess.run(accuracy, feed_dict=test_feed)
        #print("test accuracy: " + accuracy_score)
        #print(sess.run('layer1/weights:0'))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    #get_variables()
    eval_testset(mnist)



if __name__ == '__main__':
    main()
