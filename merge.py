#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author danebrown
# @date 2023/5/14
# @file 合并

import os


def merge_all_ts_file(save_temporary_ts_path,save_mp4_path):
    print('开始合并视频……')
    ts_file_list = [file for file in os.listdir(save_temporary_ts_path) if file.endswith(".fs")]
    # sorted(ts_file_list,key=lambda x: int(x.split(".")[2]))
    ts_file_list.sort(key=lambda x: int(x.split(".")[-2]))
    with open(save_mp4_path+'/video3.mp4', 'wb+') as fw:
        for i in range(len(ts_file_list)):

            fr = open(os.path.join(save_temporary_ts_path, ts_file_list[i]), 'rb')
            fw.write(fr.read())
            fr.close()
    # shutil.rmtree(save_temporary_ts_path) #删除所有的ts文件
    print('视频合并完成！')

merge_all_ts_file('/Users/baodan/develop/git/tensorflowlearn/download/视频20230514_172852','/Users/baodan/develop/git/tensorflowlearn/download')