import os

import requests
from Crypto.Cipher import AES
from Crypto.library.cryptor import Cryptor

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.56",
    "Connection": "close"
}


def m3u8(url, movie_name):
    base_url = url[:url.rfind('/') + 1]  # 用于拼接url
    rs = requests.get(url, headers=headers).text
    list_content = rs.split('\n')
    player_list = []
    # 创建文件夹，用于存放ts文件
    if not os.path.exists('{}'.format(movie_name)):
        # os.system('mkdir merge')
        os.mkdir('{}'.format(movie_name))
    key = ''
    for index, line in enumerate(list_content):
        # 判断视频是否经过AES-128加密
        if "#EXT-X-KEY" in line:
            method_pos = line.find("METHOD")
            comma_pos = line.find(",")
            method = line[method_pos:comma_pos].split('=')[1]  # 获取加密方式
            print("Decode Method：", method)
            uri_pos = line.find("URI")
            quotation_mark_pos = line.rfind('"')
            key_path = line[uri_pos:quotation_mark_pos].split('"')[1]
            key_url = key_path
            res = requests.get(key_url, headers=headers)
            key = res.content  # 获取加密密钥

        # print("key：", key)

        """
        获取.ts文件链接地址方式可根据需要进行定制
        """
        if '#EXTINF' in line:
            # 获取每一媒体序列的.ts文件链接地址
            if 'https' in list_content[index + 1]:
                href = list_content[index + 1]
                player_list.append(href)
            else:
                href = base_url + list_content[index + 1]
                player_list.append(href)
    if (len(key)):
        print('此视频经过加密')
        # print(player_list)#打印ts地址列表
        for i, j in enumerate(player_list):
            if not os.path.exists('{}/'.format(movie_name + str(i + 1) + '.ts')):
                cryptor = AES.new(key, AES.MODE_CBC, key)
                res = requests.get(j, headers=headers)
                requests.adapters.DEFAULT_RETRIES = 5
                with open('{}/'.format(movie_name) + str(i + 1) + '.ts', 'wb') as file:
                    file.write(cryptor.decrypt(res.content))  # 将解密后的视频写入文件
                    print('正在写入第{}个文件'.format(i + 1))
                    # time.sleep(5)
            else:
                # print(i)
                pass
    else:
        print('此视频未加密')
        # print(player_list)#打印ts地址列表
        for i, j in enumerate(player_list):
            if not os.path.exists('{}/'.format(movie_name + str(i + 1) + '.ts')):
                res = requests.get(j, headers=headers)
                with open('{}/'.format(movie_name) + str(i + 1) + '.ts', 'wb') as file:
                    file.write(Cryptor.decrypt(res.content))  # 将解密后的视频写入文件
                    print('正在写入第{}个文件'.format(i + 1))
    print('下载完成')


name = 'nz'
url = "https://streamobs.yunxuetang.cn/orgsv2/knowledge/18320925151/knowledgefiles/videos/202302/82e0c6a66e74413797965e905c52e3a0_3103422144_720p_enc.m3u8"
m3u8(url, name)
