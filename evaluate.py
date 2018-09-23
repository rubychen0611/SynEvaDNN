import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from abc import abstractmethod
import numpy as np
import gen_ML_model as ml
import knowledge_program as kp
import mirror_program as mp
# 测量结果保存目录
evaluate_outputs_dir = "evaluate/"

# 抽象基类：评价器
class Evaluator:
    def __init__(self, testset):
        self.testset = testset

        self.result_np = np.zeros(shape=(testset.num_examples, 2))
        #print(self.result_np)

    def calc_prediction_correctness(self):
        y_ = np.vstack([label.reshape(-1, ) for label in self.testset.labels])
        y = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, "final_outputs"), delimiter=',')
        correct_prediction = 1 * np.equal(np.argmax(y, 1), np.argmax(y_, 1))
        self.result_np[:,1] = correct_prediction
        # print(correct_prediction)

    def save_results(self, filename):
        np.savetxt(fname=os.path.join(evaluate_outputs_dir, filename + ".csv"), X=self.result_np, delimiter=',',fmt=["%.8f", "%d"])

    @abstractmethod
    def evaluate(self): pass

# 欧氏距离评价器
class EuclideanDistance(Evaluator):
    def __init__(self, testset):
        super().__init__(testset)

    def accumulate(self, knowledge_np, mirror_np, weights):
        for j in range(0, knowledge_np.shape[0]):
            for k in range(0, knowledge_np.shape[1]):
                self.result_np[j][0] += weights * np.square(knowledge_np[j][k] - mirror_np[j][k])

    def statistic_analysis(self):
        granularity = 100
        dtype = [('distance', float), ('correctness', int)]
        result_copy = np.array([(self.result_np[i][0], self.result_np[i][1]) for i in range(0, self.result_np.shape[0])],dtype=dtype)
        print(result_copy)

        result_sorted = np.sort(result_copy, axis=0, kind='quicksort', order='distance')
        print(result_sorted)

        sum = 0
        for i in range(0, self.result_np.shape[0]):
            sum += result_sorted[i][1]
            if (i+1) % granularity == 0:
                print(sum)
                sum = 0



    def evaluate(self):
        # 计算结果是否正确
        self.calc_prediction_correctness()
        # 累加计算欧式距离
        for i in range(1, ml.HIDDEN_LAYER_NUM + 1):
            layer_name = 'hidden_layer_' + str(i)
            knowledge_np = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, layer_name),delimiter=',')
            mirror_np = np.loadtxt(os.path.join(mp.mirror_outputs_dir, layer_name),delimiter=',')
            self.accumulate(knowledge_np, mirror_np, i)
        knowledge_np = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, "final_outputs"), delimiter=',')
        mirror_np = np.loadtxt(os.path.join(mp.mirror_outputs_dir, "final_outputs"), delimiter=',')
        self.accumulate(knowledge_np, mirror_np, ml.HIDDEN_LAYER_NUM + 1)
        for i in range(0, self.result_np.shape[0]):
           self.result_np[i][0] = np.sqrt(self.result_np[i][0])

        # 统计分析
        self.statistic_analysis()
        # 保存结果
        # self.save_results(self.__class__.__name__)





def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    evaluator = EuclideanDistance(mnist.validation)
    evaluator.evaluate()

if __name__ == '__main__':
    main()