import base64
import json
import uuid
import requests
import time
import os
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from io import BytesIO
from .audio_player import AudioPlayer

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
        self.vad = None  # 添加 VAD 引用
        self.audio_player = AudioPlayer()

    def set_vad(self, vad):
        """设置 VAD 实例"""
        self.vad = vad
        self.audio_player.vad = vad
        
    def synthesize(self, text: str) -> str:
        """
        将文本转换为语音并返回音频文件路径
        Args:
            text: 要转换的文本
        Returns:
            str: 音频文件路径
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
                    "encoding": "wav",
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
                return None
                
            resp_json = resp.json()
            
            if "data" not in resp_json:
                print(f"TTS响应中没有音频数据: {resp_json}")
                return None
            
            # 解码音频数据
            audio_data = base64.b64decode(resp_json["data"])
            
            # 计算合成耗时
            synthesis_time = (time.time() - start_time) * 1000
            print(f"合成完成：{text} | 耗时: {synthesis_time:.2f}ms")
            
            # 保存音频文件到临时目录
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, f"tts_{int(time.time() * 10)}.wav")
            
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            # 返回音频文件路径而不是直接播放
            return temp_file
            
        except Exception as e:
            print(f"语音合成出错: {str(e)}")
            return None

    def play_audio(self, file_path):
        """使用sounddevice播放音频文件"""
        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"播放音频失败: {e}")


if __name__ == "__main__":
    tts = TTS()
    tts.synthesize("你好，我是你的语音小助手~")

