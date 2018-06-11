# autoregressive-lstm-for-time-series-analysis
Long short-term memory (LSTM) model extended for long time-series autoregression problem

通过自回归长短期记忆模型将时间序列数据转化为向量表示。充分利用时间序列自身变化规律，以非监督的方式提取时间序列数据的时变特征，得到各时点系统的长期记忆和短期状态。所得的长期和短期特征向量可用于后续分类、回归等任务（数据标签可能数量有限）

This is an implementation of autoregressive lstm for encoding distributed representation of a time-series. The model learns long-term and short-term states of the sequence at each time point in a unsupervised manner with full utilization of the inherent pattern in the time-series. The long-term and short-term features can be used for downstream tasks such as classification and regression, which may be performed with rather limited available labels.


应用案例1：提取身体信号特征，将长期特征用于预测用户性格倾向

Ex 1: decoding body signal into feature vector for personality prediction

![image](https://github.com/wenq10/autoregressive-lstm-for-time-series-analysis/blob/master/figures/corr.png)

![image](https://github.com/wenq10/autoregressive-lstm-for-time-series-analysis/blob/master/figures/pdct.png)


应用案例2：提取身体信号特征，将短期特征用于预测用户即时感知压力

Ex 2: mining body signals for predicting instantaneous stress level

![image](https://github.com/wenq10/autoregressive-lstm-for-time-series-analysis/blob/master/figures/stress.png)
