#!/usr/bin/env python

import os

def gen_txt(im_dir, txt_path):
    abs_im_dir = os.path.abspath(im_dir)
    print('abs_im_dir is:', abs_im_dir)
    fout = open(txt_path, 'w')
    for item in os.listdir('test'):
        if item.endswith('.bmp'):
            fullpath = os.path.join(abs_im_dir, item)
            fout.write(fullpath+'\n')
    fout.close()

gen_txt('train', 'train_img_list.txt')
gen_txt('test', 'test_img_list.txt')
