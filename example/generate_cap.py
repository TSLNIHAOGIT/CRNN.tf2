# coding:utf-8

from captcha.image import ImageCaptcha
import numpy as np
# import cv2
import random, os, sys
from random import shuffle

def shuffle_str(s):
    # 将字符串转换成列表
    str_list = list(s)
    # 调用random模块的shuffle函数打乱列表
    shuffle(str_list)
    # 将列表转字符串
    return ''.join(str_list)

class LstmCtcOcr:
    def __init__(self):

        self.im_width = 256
        self.im_height = 64
        self.im_total_num = 100  # (每种组合生成验证码的数量)总共生成的验证码图片数量
        self.train_max_num = self.im_total_num  # 训练时读取的最大图片数目

        self.words_max_num = 5  # 每张验证码图片上的最大字母个数
        self.words = shuffle_str('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        self.n_classes = len(self.words) + 1  # 26个字母 + blank

    def createCaptchaDataset(self):
        """
        生成训练用图片数据集
        :return:
        """
        image = ImageCaptcha(width=self.im_width, height=self.im_height, font_sizes=(56,))



        for k in range(1,self.words_max_num):
            # print('k', k)
            # 每种组合生成的次数
            # for n in range(1, 5):
                    n=k
                    if n==1:
                        self.im_total_num=100
                    elif n==2:
                        self.im_total_num = self.im_total_num*2
                    elif n==3:
                        self.im_total_num = self.im_total_num*4
                    elif n==4:
                        self.im_total_num = self.im_total_num*8
                    print('self.im_total_num',self.im_total_num)
                    for i in range(self.im_total_num):
                        words_tmp = ''

                        for j in range( k):
                            words_tmp = words_tmp + random.choice(self.words)
                        print(words_tmp, type(words_tmp))
                        base_path = r'../all_datas'
                        im_path = '../all_datas/captcha_datas/%d_%s.png' % (i, words_tmp)

                        # print(os.path.exists(base_path), im_path)
                        # image.write(words_tmp, im_path)
if __name__=='__main__':
    LstmCtcOcr().createCaptchaDataset()