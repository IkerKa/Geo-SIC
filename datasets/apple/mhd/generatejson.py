import SimpleITK as sitk
import os, glob
import json
import numpy as np
import csv

with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    your_list = list(reader)
np.array(your_list)
df_n = np.array(your_list)
keyword = 'train'
dictout = {keyword:[]}

dataset_name = 'image'
dataset_path = 'datasets/apple/mhd'
#len of the csv file
elements = len(df_n)
for i in range(0, elements):
	smalldict = {}
	filename = dataset_path + '/' + dataset_name + '_' + str(i) + '.mhd'
	smalldict['image'] = filename
	smalldict ['label'] = df_n[i][0]
	dictout[keyword].append(smalldict)
	print (filename, df_n[i][0])

savefilename = './data'+ '.json'
with open(savefilename, 'w') as fp:
	json.dump(dictout, fp)