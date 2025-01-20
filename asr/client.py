import requests
import time
import os
from typing import Optional
from dotenv import load_dotenv

class ASRClient:
    def __init__(self, server_url: str = None):
        """
        初始化ASR客户端
        Args:
            server_url: ASR服务器地址，如果为None则从环境变量读取
        """
        # 加载环境变量
        load_dotenv()
        
        self.server_url = server_url or os.getenv("ASR_SERVER_URL", "http://localhost:49180")
        print(f"ASR客户端初始化完成，服务器地址: {self.server_url}")

    def transcribe(self, audio_path: str, batch_size: int = 64) -> Optional[str]:
        """
        将音频文件转换为文本
        Args:
            audio_path: 音频文件路径
            batch_size: 批处理大小
        Returns:
            识别结果文本
        """
        try:
            # 开始计时
            start_time = time.time()
            
            # 直接读取音频文件并发送
            with open(audio_path, 'rb') as f:
                files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
                params = {'batch_size': batch_size}
                response = requests.post(
                    f"{self.server_url}/transcribe",
                    files=files,
                    params=params,
                    stream=True  # 使用流式传输
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        print(f"转录耗时: {result['elapsed_time']:.2f}ms | 总耗时: {(time.time() - start_time) * 1000:.2f}ms")
                        print(f"识别结果: {result['text']}")
                        return result['text']
                    else:
                        print(f"识别失败: {result.get('message', '未知错误')}")
                        return None
                else:
                    print(f"服务器错误: {response.text}")
                    return None
            
        except Exception as e:
            print(f"转录过程出现错误: {str(e)}")
            return None


if __name__ == "__main__":
    # 测试代码
    asr = ASRClient()
    text = asr.transcribe("vad/output/output_1737293228_0.wav", batch_size=64)

