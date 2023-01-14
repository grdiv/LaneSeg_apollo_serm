'''
Author: dfs
Date: 2023-01-01 21:31:38
LastEditors: dfs
LastEditTime: 2023-01-14 13:11:13
Description: dfs
'''

import pandas as pd
import os
from sklearn.model_selection import train_test_split

img_list = []
train_list,val_list,test_list = [],[],[]

img_data_root = '/media/dfs/Samsung_T5/dataset/SeRM/train'
label_data_root = '/media/dfs/Samsung_T5/dataset/SeRM/trainannot'

for img_name in os.listdir(img_data_root):
    label_name = img_name.replace('image', 'annotation')
    label_path = os.path.join(label_data_root, label_name)
    img_path = os.path.join(img_data_root, img_name)
    img_list.append([img_path, label_path])

train_list, val_list = train_test_split(img_list, test_size=0.1, shuffle=True)

train_part = pd.DataFrame(train_list, columns=['Image','Label'])
val_part = pd.DataFrame(val_list, columns=['Image','Label'])

train_part.to_csv('data_list/train_lite.csv')
val_part.to_csv('data_list/val_lite.csv')

pass