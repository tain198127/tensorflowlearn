import requests
header={
"authority": "streamobs.yunxuetang.cn",
"method": "GET",
"scheme": "https",
"accept": "*/*",
"accept-encoding": "gzip, deflate, br",
"accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
"if-modified-since": "Fri, 17 Feb 2023 04:42:46 GMT",
"if-none-match": "3d7f2cae10289c5186468576c2701077",
"origin": "http://sunline.yunxuetang.cn",
"referer": "http://sunline.yunxuetang.cn/",
"sec-ch-ua": '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
"sec-ch-ua-mobile": "?0",
"sec-ch-ua-platform": "macOS",
"sec-fetch-dest": "empty",
"sec-fetch-mode": "cors",
"sec-fetch-site": "cross-site",
"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/112.0.0.0 Safari/537.36"
}


#
# def download(url):
#     response = requests.get(url=url, headers=header)
#     if response.status_code == 200:
#         return response
# def merge():
#     pass
#
# def encrypt(data):
#     //{"videoKeyId":"job-pbghpve08hxiigek","playerId":"pid-1-5-1","encryptedVideoKey":"a3acdbfdebc7d52a5cf461602e6d471c"}
#     pass
#
# if __name__ == '__main__':
#     url = "https://streamobs.yunxuetang.cn/orgsv2/knowledge/18320925151/knowledgefiles/videos/202302" \
#           "/82e0c6a66e74413797965e905c52e3a0_3103422144_720p_enc.m3u8.{}.ts"
#     for i in range(0,610):
#         tsurl = url.format(i)
#         file = download(tsurl).content
#         encrypt(file)

import requests
from Crypto.Cipher import AES
import binascii


def download_encrypted_video(urls, key, iv):
    # 发送请求获取加密视频的索引文件内容


    # 解析索引文件，提取加密的视频片段链接
    segments = urls



    # 下载并解密每个视频片段
    for i, segment_url in enumerate(segments):
        response = requests.get(segment_url)
        encrypted_data = response.content

        # 使用AES-128解密视频片段
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(encrypted_data)

        # 保存解密后的视频片段到本地文件
        with open(f'segment_{i}.ts', 'wb') as file:
            file.write(decrypted_data)


# 测试程序
playlist_url = 'https://streamobs.yunxuetang.cn/orgsv2/knowledge/18320925151/knowledgefiles/videos/202302/82e0c6a66e74413797965e905c52e3a0_3103422144_720p_enc.m3u8'
key_uri = 'https://drm.media.baidubce.com/v1/tokenVideoKey?videoKeyId=job-pbghpve08hxiigek'
iv = '0x1e7322cc1101980e5ab2a4f6e26e7e87'
encryptedVideoKey = "a3acdbfdebc7d52a5cf461602e6d471c"
seg = []
for i in range(610):
    seg.append('https://streamobs.yunxuetang.cn/orgsv2/knowledge/18320925151/knowledgefiles/videos/202302' \
             '/82e0c6a66e74413797965e905c52e3a0_3103422144_720p_enc.m3u8.{}.ts'.format(i))
    key =bytes(encryptedVideoKey, encoding='utf-8')
    IV = bytes(iv, encoding='utf-8')
download_encrypted_video(seg, key, IV)
