\
* When I spilt every function, I met a problems:

evaluation:
  ranker = globals()[model_id]() 
  globals of function

the evaluation can not find the model_id(), so i fix now


* torch.sum()对输入的tensor数据的某一维度求和，一共两种用法
* torch.sum(_batch_loss, dim=1): Returns the sum of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.
if dim=1, return the sum of dim=1, and reduce dim=1.
* relevance_preds,std_labels: torch.Size([12, 8])
* super() 函数  super(RankMSE,self) 首先找到 RankMSE 的父类（就是类 NeuralRanker），然后把类 RankMSE 的对象转换为类 NeuralRanker 的对象
* reduction :

  - ‘none’:no reduction will be applied.按照原始维度输出
  - ‘mean’: the sum of the output will be divided by the number of elements in the output.
  - ‘sum’: the output will be summed.
  
