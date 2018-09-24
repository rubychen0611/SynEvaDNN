from evaluators.evaluator import Evaluator
import os
import numpy as np
import gen_ML_model as ml
import knowledge_program as kp
import mirror_program as mp

# 欧氏距离评价器
class EuclideanDistance(Evaluator):
    def __init__(self, testset, if_use_weights = True):
        super().__init__(testset)
        self.if_use_weights = if_use_weights

    def accumulate_with_weights(self, knowledge_np, mirror_np, weights):
        '''
        累加差的平方, 使用加权
        '''
        for j in range(0, knowledge_np.shape[0]):
            for k in range(0, knowledge_np.shape[1]):
                self.result_np[j][0] += weights * np.square(knowledge_np[j][k] - mirror_np[j][k])

    def accumulate(self, knowledge_np, mirror_np):
        '''
        累加差的平方, 不使用加权
        '''
        for j in range(0, knowledge_np.shape[0]):
            for k in range(0, knowledge_np.shape[1]):
                self.result_np[j][0] += np.square(knowledge_np[j][k] - mirror_np[j][k])

    def evaluate(self):
        # 计算结果是否正确
        self.calc_prediction_correctness()
        # 累加计算欧式距离
        for i in range(1, ml.HIDDEN_LAYER_NUM + 1):
            layer_name = 'hidden_layer_' + str(i)
            knowledge_np = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, layer_name),delimiter=',')
            mirror_np = np.loadtxt(os.path.join(mp.mirror_outputs_dir, layer_name),delimiter=',')
            if self.if_use_weights:
                self.accumulate_with_weights(knowledge_np, mirror_np, i)
            else:
                self.accumulate(knowledge_np, mirror_np)
        knowledge_np = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, "final_outputs"), delimiter=',')
        mirror_np = np.loadtxt(os.path.join(mp.mirror_outputs_dir, "final_outputs"), delimiter=',')
        if self.if_use_weights:
            self.accumulate(knowledge_np, mirror_np, ml.HIDDEN_LAYER_NUM + 1)
        else:
            self.accumulate(knowledge_np, mirror_np)
        for i in range(0, self.result_np.shape[0]):
           self.result_np[i][0] = np.sqrt(self.result_np[i][0])

        # 统计分析
        # self.statistic_analysis_equal_interval(granularity=500)
        self.statistic_analysis_group_count(intervals=[1,2,3,4])
        # 保存结果
        # self.save_results(self.__class__.__name__)