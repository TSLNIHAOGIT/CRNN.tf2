import os.path as osp
from tqdm import tqdm
# import random
import os
import re
import numpy as np
# def create_dataset_from_dir(root):
#     img_names = os.listdir(root)
#     print('img_names',img_names)
#     img_paths = []
#     for img_name in tqdm(img_names, desc="read dir:"):
#         img_name = img_name.rstrip().strip()
#         img_path = root + "/" + img_name
#         if osp.exists(img_path):
#             img_paths.append(img_path)
#     labels = [img_path.split("/")[-1].split("_")[-1].split('.')[0] for img_path in tqdm(img_paths, desc="generator label:")]
#     return img_paths, labels
def get_annoation(path_img=None,path_label=None):

    path=path_img
    path2=path_label

    file_name = os.listdir(path)
    print('file_name', file_name)
    np.random.seed(0)
    np.random.shuffle(file_name)
    amounts=len(file_name)
    print('file_name',file_name)

    for index,each_name in enumerate(file_name):
        # print(each_name)
        abs_path = os.path.join(path, each_name)
        notation = re.split('\_|\.', each_name)[1]
        # print('each_name={}, notation={}'.format(each_name, notation))
        if index<=int(0.8*amounts):
            with open(os.path.join(path2, 'annotation_train.txt'), 'a+') as f:
                f.write('{} {}\n'.format('{}'.format(abs_path), notation))
        elif index<=int(0.9*amounts):
            with open(os.path.join(path2, 'annotation_val.txt'), 'a+') as f:
                f.write('{} {}\n'.format('{}'.format(abs_path), notation))
        else:
            with open(os.path.join(path2, 'annotation_test.txt'), 'a+') as f:
                f.write('{} {}\n'.format('{}'.format(abs_path), notation))
        print(abs_path,notation)






if __name__=='__main__':
    # root=r'E:\tsl_file\python_project\CRNN.tf2'
    # file_path=r'example\images'
    # res=create_dataset_from_file(root, file_path)
    # print('res:\n',res)
    # path=r'E:\tsl_file\python_project\CRNN.tf2\example\images'
    # res=create_dataset_from_dir(path)
    # print('res\n',res[0],'\n',res[1])


    path = r'E:\tsl_file\python_project\CRNN.tf2\example\images'
    path2 = r'E:\tsl_file\python_project\CRNN.tf2'

    get_annoation(path,path2)