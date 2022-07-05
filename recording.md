\
* When I spilt every function by wirting myself, I met a problems:
evaluation:
  ranker = globals()[model_id]() 
  globals of function

  the evaluation can not find the model_id(), so I fix it now


* torch.sum()对输入的tensor数据的某一维度求和，一共两种用法
* torch.sum(_batch_loss, dim=1): Returns the sum of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.
if dim=1, return the sum of dim=1, and reduce dim=1.
* relevance_preds,std_labels: torch.Size([12, 8])
* super() 函数  super(RankMSE,self) 首先找到 RankMSE 的父类（就是类 NeuralRanker），然后把类 RankMSE 的对象转换为类 NeuralRanker 的对象
* reduction :

  - ‘none’:no reduction will be applied.按照原始维度输出
  - ‘mean’: the sum of the output will be divided by the number of elements in the output.
  - ‘sum’: the output will be summed.
  
* torch.gather:
  * 定义：从原tensor中获取指定dim和指定index的数据
  * 如果dim=0，则b相对于a，它存放的都是第0维度的索引； 
  * 如果dim=1，则b相对于a，它存放的都是第1维度的索引； 
  * 我举个栗子，当dim=0时，b[0][0]的元素是1，那么它想要查找a[0][1]中的元素; 
  * 当dim=1时，b[0][0]的元素是1，那么它想查找的a[1][0]中的元素；
  * 最后的输出都可以看作是对a的查询，即元素都是a中的元素，查询索引都存在b中。输出大小与b一致。
* torch.sort:
  * 对tensor中元素排序
  * dim= 1按照列排序
  * 返回值有两个：第一个是排序后的数据，第二个是排序后的索引
  * 同时排序 排序的数据下标

### question1: torch.gateher->torch.sort ?
 #batch_desc_stds, batch_desc_stds_inds= torch.sort(batch_std_labels, dim=1,descending=True)
