from tensorflow.examples.tutorials.mnist import input_data
from evaluators.activated_neuron_similarity import ActivatedNeuronSimilarity
from evaluators.eculidean_distance import EuclideanDistance


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    evaluator = EuclideanDistance(mnist.validation, if_use_weights=False)
    #evaluator = ActivatedNeuronSimilarity(mnist.validation)
    evaluator.evaluate()

if __name__ == '__main__':
    main()