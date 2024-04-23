## PIST

This is an implementation of the paper
Teng Huang, Bin-Bin Jia, Min-Ling Zhang. Deep Multi-Dimensional Classification with Pairwise Dimension-Specific Features. In: Proceedings of the 33th International Joint Conference on Artificial Intelligence (IJCAI'24), Jeju.

Github link: https://github.com/tengingit/PIST
***

## Requirements

- Python == 3.9.18
- Pytorch == 1.12.1
- numpy == 1.26.0
***

### Datasets

All data sets can be downloaded from https://palm.seu.edu.cn/zhangml/Resources.htm#MDC_data.

### Train and Test

For example, to perform 10-fold cross validation on *BeLaE* data set:

```
python main.py -dataset BeLaE
```

We keep the training log on **'logs'** directory and testing results on **'results'** directory.






