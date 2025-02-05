import SimpleITK as sitk
import os, glob
import json
import numpy as np
import csv
import argparse

def main(label_folder):
    label_file = os.path.join(label_folder, 'label.csv')
    with open(label_file, newline='') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    np.array(your_list)
    df_n = np.array(your_list)
    keyword = 'train'
    dictout = {keyword:[]}

    dataset_name = 'image'
    dataset_path = 'datasets/' + args.label_folder
    #len of the csv file
    elements = len(df_n)
    for i in range(0, elements):
        smalldict = {}
        filename = dataset_path + '/' + dataset_name + '_' + str(i) + '.mhd'
        smalldict['image'] = filename
        smalldict ['label'] = df_n[i][0]
        dictout[keyword].append(smalldict)
        print (filename, df_n[i][0])

    savefilename = './'+ args.label_folder + '.json'
    with open(savefilename, 'w') as fp:
        json.dump(dictout, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate JSON from CSV labels.')
    parser.add_argument('label_folder', type=str, help='Folder where the label.csv is located')
    args = parser.parse_args()
    main(args.label_folder)