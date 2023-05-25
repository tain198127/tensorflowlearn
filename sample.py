#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author souldou-1257096360@qq.com
# @date 2021/4/1
# @file 视频检测index.py

# 用到的依赖，这里重点说一下 Crypto 的依赖 肯定会遇到问题，直接谷歌一下就能解决
import binascii
import os
import requests
import json
import datetime
from Crypto.Cipher import AES


class Reptile(object):
    def __init__(self, params: object):
        self.url, self.token, self.data,self.IV = params['url'], params['token'], params['data'],params["IV"]
        print('Reptile!!!,\nurl:{},\ndata:{}\n'.format(self.url, self.data))
        self.header = {}  # 自己的请求头信息 自己模拟
        # 创建问价夹
        download_path = os.getcwd() + "/download"
        if not os.path.exists(download_path):
            os.mkdir(download_path)
        # 新建日期文件夹
        download_path = os.path.join(download_path, '视频' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.mkdir(download_path)
        all_content = requests.get(url=self.url, headers=self.header).text  # 获取M3U8的文件内容
        # 保存m3u8文件为文本
        with open(download_path + "\m3u8.txt", 'w+') as f:
            f.write(all_content)
            f.close()
        file_line = all_content.split("\n")  # 读取文件里的每一行
        # 通过判断文件头来确定是否是M3U8文件
        if file_line[0] != "#EXTM3U":
            raise BaseException(u"非M3U8的链接")
        else:
            unknow = True  # 用来判断是否找到了下载的地址
            for index, line in enumerate(file_line):
                unknow = False
                if "#EXT-X-KEY" in line:
                    # 找m3u8文件中的参数和解密的key,请求连接
                    URI_http = line[line.find("URI"):line.rfind('"')].split('"')[1]
                    # 找m3u8文件中的参数和解密的IV
                    # IV = line[line.find("IV"):].split(',')[0].split('=')[1]
                    # IV=binascii.unhexlify('1e7322cc1101980e5ab2a4f6e26e7e87')
                    IV=binascii.unhexlify(self.IV)
                    # 拿到请求返回的真实key
                    tokenVideoKey = requests.get(
                        url=URI_http + '&playerId=' + '&playerId=pid-1-5-1&token=' + self.token)
                    data = json.loads(tokenVideoKey.content)
                    # encryptedVideoKey 就是借口返回的key
                    encryptedVideoKey, videoKeyId = data['encryptedVideoKey'], data[
                        'videoKeyId']

                    # 开始解密文件和保存
                    # 拼出ts片段的URL，拼接处ts文件的请求地址：https://jh0p4t0rh9rs9610ryc.exp.bcevod.com/mda-kimg4q6rjx1351mk/mda-kimg4q6rjx1351mk.m3u8.15.ts 这样的格式（此链接无效）
                    pd_url = self.url.rsplit("/", 1)[0] + "/" + file_line[index + 2]
                    # 获取ts视频文件
                    res = requests.get(pd_url)
                    print('res:', res)
                    c_fule_name = str(file_line[index + 2])
                    print(c_fule_name + ".mp4")
                    if encryptedVideoKey and len(encryptedVideoKey):
                        # AES 解密
                        KEY = bytes(encryptedVideoKey, encoding='utf-8')
                        # IV = bytes(IV, encoding='utf-8')

                        cont = res.content

                        print('key:{}\nIV:{}\n'.format(KEY, IV))

                        crypto = AES.new(KEY, AES.MODE_CBC, IV)
                        c_fule_name + ".mp4"
                        with open(os.path.join(download_path, c_fule_name), 'ab') as f:
                            # f.write(crypto.decrypt(cont))
                            f.write(cont)
                            pass
                        pass
                    else:
                        print('AES 解密跳过')
                        with open(os.path.join(download_path, c_fule_name), 'ab') as f:
                            f.write(res.content)
                            f.flush()
                            pass
                        pass
                    pass
                pass
            if unknow:
                raise BaseException("未找到对应的下载链接")
            else:
                print("下载完成")
            pass
        pass


if __name__ == '__main__':
    # 这里的url 自己在浏览器控制台复制 .m3u8 的请求地址，token获取 下边看截图，会在/v1/tokenVideoKey?的这个接口中获取
    params = {
        "data": {},
        "url": 'https://streamobs.yunxuetang.cn/orgsv2/knowledge/18320925151/knowledgefiles/videos/202302/82e0c6a66e74413797965e905c52e3a0_3103422144_720p_enc.m3u8',
        "token": '332df909fd332318c1392338c225aebc0c21fbf703526f9f6e0162d03359e88f_7d2195c92f8842a586f4299a8244b1fa_1684069780',
        "IV":"1e7322cc1101980e5ab2a4f6e26e7e87"
    }
    reptile = Reptile(params)

