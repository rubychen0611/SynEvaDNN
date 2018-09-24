import os
from abc import abstractmethod
import numpy as np
import knowledge_program as kp

# 测量结果保存目录
evaluate_outputs_dir = "outputs/evaluate_results/"

# 抽象基类：评价器
class Evaluator:
    def __init__(self, testset):
        self.testset = testset
        self.result_np = np.zeros(shape=(testset.num_examples, 2))

    def calc_prediction_correctness(self):
        y_ = np.vstack([label.reshape(-1, ) for label in self.testset.labels])
        y = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, "final_outputs"), delimiter=',')
        correct_prediction = 1 * np.equal(np.argmax(y, 1), np.argmax(y_, 1))
        self.result_np[:,1] = correct_prediction

    def save_results(self, filename):
        np.savetxt(fname=os.path.join(evaluate_outputs_dir, filename + ".csv"), X=self.result_np, delimiter=',',fmt=["%.8f", "%d"])


    def statistic_analysis_equal_interval(self, granularity):
        '''
        统计将 距离/相似度 排序后，每个小区间内Knowledge Program预测错误的个数
        '''
        dtype = [('distance', float), ('correctness', int)]
        result_copy = np.array([(self.result_np[i][0], self.result_np[i][1]) for i in range(0, self.result_np.shape[0])],dtype=dtype)
        #print(result_copy)

        result_sorted = np.sort(result_copy, axis=0, kind='quicksort', order='distance')
        #print(result_sorted)

        sum = 0
        for i in range(0, self.result_np.shape[0]):
            sum += result_sorted[i][1]
            if (i + 1) % granularity == 0:
                print(sum)
                sum = 0

    def statistic_analysis_group_count(self, intervals):
        dtype = [('distance', float), ('correctness', int)]
        result_copy = np.array(
            [(self.result_np[i][0], self.result_np[i][1]) for i in range(0, self.result_np.shape[0])], dtype=dtype)
        # print(result_copy)

        result_sorted = np.sort(result_copy, axis=0, kind='quicksort', order='distance')
        # print(result_sorted)

        k = 0
        sum = 0
        count = 0
        for i in range(0, result_sorted.shape[0]):
            if k < len(intervals) and result_sorted[i][0] > intervals[k]:
                print(sum / count)
                sum = 0
                count = 0
                k += 1
            sum += result_sorted[i][1]
            count += 1
        print(sum / count)

    @abstractmethod
    def evaluate(self):
        '''抽象方法：评价'''
        pass
