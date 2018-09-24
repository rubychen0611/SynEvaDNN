import os
import numpy as np
import mirror_program as mp
import knowledge_program as kp
import gen_ML_model as ml
from evaluators.evaluator import Evaluator

# 被激活神经元相似度评价器
class ActivatedNeuronSimilarity(Evaluator):
    def __init__(self, testset):
        super().__init__(testset)

    def accumulate(self, knowledge_np, mirror_np, temp, t):
        for j in range(0, knowledge_np.shape[0]):
            for k in range(0, knowledge_np.shape[1]):
                a = knowledge_np[j][k]
                b = mirror_np[j][k]
                if a > t and b > t:
                    temp[j][0] += 1
                    temp[j][1] += 1
                elif (a > t and b <= t) or (a <= t and b > t):
                    temp[j][1] += 1

    def evaluate(self):
        # 计算结果是否正确
        self.calc_prediction_correctness()

        # 计算被激活神经元相似度
        t = 1 # 设置阈值t
        temp = np.zeros(shape=(self.testset.num_examples, 2), dtype=int)
        for i in range(1, ml.HIDDEN_LAYER_NUM + 1):
            layer_name = 'hidden_layer_' + str(i)
            knowledge_np = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, layer_name), delimiter=',')
            mirror_np = np.loadtxt(os.path.join(mp.mirror_outputs_dir, layer_name), delimiter=',')
            self.accumulate(knowledge_np, mirror_np, temp, t)
        knowledge_np = np.loadtxt(os.path.join(kp.knowledge_outputs_dir, "final_outputs"), delimiter=',')
        mirror_np = np.loadtxt(os.path.join(mp.mirror_outputs_dir, "final_outputs"), delimiter=',')
        self.accumulate(knowledge_np, mirror_np, temp, t)
        #print(temp)
        for i in range(0, self.result_np.shape[0]):
            self.result_np[i][0] = temp[i][0] / temp[i][1]

        self.statistic_analysis(granularity=500)
        # 保存结果
        self.save_results(self.__class__.__name__)