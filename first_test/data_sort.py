#@inproceedings{liu2015faceattributes,
# author = {Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang},
# title = {Deep Learning Face Attributes in the Wild},
# booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
# month = December,
# year = {2015}
#}

import pandas as pd

df = pd.read_csv('dataset/list_attr_celeba.txt', delimiter=r"\s+",skiprows=1)
print(df)

data_ans = df[['Male']]
data_non_gender_attri = df.drop(['Male'],axis=1)

data_ans.to_csv('dataset/data_gender.csv', index = False)
data_non_gender_attri.to_csv('dataset/data_other_attri.csv', index = False)
