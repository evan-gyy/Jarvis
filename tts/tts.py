import base64
import json
import uuid
import requests
import time
import os
from dotenv import load_dotenv
import pygame
from io import BytesIO

class TTS:
    def __init__(self):
        """初始化TTS类"""
        # 加载环境变量
        load_dotenv()
        
        # 从环境变量获取配置
        self.appid = os.getenv("BYTEDANCE_APPID")
        self.access_token = os.getenv("BYTEDANCE_ACCESS_TOKEN")
        self.cluster = os.getenv("BYTEDANCE_CLUSTER_TTS")
        self.voice_type = os.getenv("BYTEDANCE_VOICE_TYPE")
        
        if not all([self.appid, self.access_token, self.cluster, self.voice_type]):
            raise ValueError("请在.env文件中设置所有必需的字节跳动TTS配置")
        
        self.host = "openspeech.bytedance.com"
        self.api_url = f"https://{self.host}/api/v1/tts"
        self.header = {"Authorization": f"Bearer;{self.access_token}"}
        
        # 初始化pygame音频
        pygame.mixer.init()

    def synthesize(self, text: str) -> bool:
        """
        将文本转换为语音并播放
        Args:
            text: 要转换的文本
        Returns:
            bool: 是否成功
        """
        try:
            # 构建请求
            request_json = {
                "app": {
                    "appid": self.appid,
                    "token": self.access_token,
                    "cluster": self.cluster
                },
                "user": {
                    "uid": "388808087185088"
                },
                "audio": {
                    "voice_type": self.voice_type,
                    "encoding": "mp3",
                    "speed_ratio": 1.0,
                    "volume_ratio": 1.0,
                    "pitch_ratio": 1.0,
                },
                "request": {
                    "reqid": str(uuid.uuid4()),
                    "text": text,
                    "text_type": "plain",
                    "operation": "query",
                    "with_frontend": 1,
                    "frontend_type": "unitTson"
                }
            }
            
            # 开始计时
            start_time = time.time()
            
            # 发送请求
            resp = requests.post(self.api_url, json.dumps(request_json), headers=self.header)
            
            if resp.status_code != 200:
                print(f"TTS API请求失败: {resp.status_code}")
                return False
                
            resp_json = resp.json()
            
            if "data" not in resp_json:
                print(f"TTS响应中没有音频数据: {resp_json}")
                return False
            
            # 解码音频数据
            audio_data = base64.b64decode(resp_json["data"])
            
            # 计算合成耗时
            end_time = time.time()
            synthesis_time = (end_time - start_time) * 1000
            print(f"语音合成耗时: {synthesis_time:.2f}ms")
            
            # 使用pygame播放音频
            audio_file = BytesIO(audio_data)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            return end_time
            
        except Exception as e:
            print(f"语音合成出错: {str(e)}")
            return False


if __name__ == "__main__":
    tts = TTS()
    tts.synthesize("你好，我是你的语音小助手~")

