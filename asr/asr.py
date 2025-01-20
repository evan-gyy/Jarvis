import os
import time
import json
import asyncio
from dotenv import load_dotenv
from .streaming_asr_demo import execute_one

class ASR:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 从环境变量获取配置
        self.appid = os.getenv("BYTEDANCE_APPID")
        self.token = os.getenv("BYTEDANCE_ACCESS_TOKEN")
        self.cluster = os.getenv("BYTEDANCE_CLUSTER_ASR")
        
        if not all([self.appid, self.token, self.cluster]):
            raise ValueError("请确保环境变量中设置了BYTEDANCE_APPID、BYTEDANCE_ACCESS_TOKEN和BYTEDANCE_CLUSTER_ASR")

    def transcribe(self, audio_path):
        """
        将音频文件转换为文本
        
        Args:
            audio_path (str): 音频文件的路径
            
        Returns:
            str: 识别出的文本，如果识别失败返回None
        """
        try:
            # 开始计时
            transcribe_start = time.time()
            
            # 判断音频格式
            audio_format = "wav" if audio_path.lower().endswith('.wav') else "mp3"
            
            # 调用字节跳动ASR服务
            result = execute_one(
                {
                    'id': 1,
                    'path': audio_path
                },
                cluster=self.cluster,
                appid=self.appid,
                token=self.token,
                format=audio_format
            )
            
            # 计算转录耗时
            # transcribe_time = (time.time() - transcribe_start) * 1000
            # print(f"转录耗时: {transcribe_time:.2f}ms")
            
            # 解析结果
            if result and 'result' in result:
                return result['result']['payload_msg']['result'][0]['text']
                    
            return None
            
        except Exception as e:
            print(f"语音识别出错: {str(e)}")
            return None


if __name__ == "__main__":
    asr = ASR()
    text = asr.transcribe("vad/output/output_1737293087_0.wav")
    print(text)

