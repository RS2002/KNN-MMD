# KNN-MMD

**Article:** KNN-MMD: Cross Domain Alignment Based on Local Distribution (under way)

![](./img/model.png)

## 1. Data

### 1.1 Dataset

Public Dataset: [WiGesture](https://paperswithcode.com/dataset/wigesture)

Proposed Dataset: WiFall (./WiFall)



### 1.2 Data Preparation

Refer to [RS2002/CSI-BERT: Official Repository for The Paper, Finding the Missing Data: A BERT-inspired Approach Against Package Loss in Wireless Sensing (github.com)](https://github.com/RS2002/CSI-BERT)



## 2. Run the model

![](./img/network.png)

To run the model, follow these instructions based on the dataset you are using. For the WiGesture Dataset, use the `train.py` script, and for the WiFall Dataset, use the `train_fall.py` script. The steps to execute them are the same, and here we provide an example using `train.py`.

```
python train.py --k <shot number> --n <neighbor number for KNN> --p <select the top p samples from testing set for MK-MMD (p<1)> --task <action or people> --lr <learning rate>
```

Make sure to replace the following placeholders with the appropriate values:

- `<shot number>`: Specify the shot number.
- `<neighbor number for KNN>`: Specify the number of neighbors for KNN.
- `<select the top p samples from testing set for MK-MMD (p<1)>`: Specify the value for p (selecting the top p samples from the testing set for MK-MMD). Note that p should be less than 1.
- `<action or people>`: Specify the task name as either "action" or "people".
- `<learning rate>`: Specify the desired learning rate.

Once you have set the appropriate values, run the command in your terminal to start the training process.
