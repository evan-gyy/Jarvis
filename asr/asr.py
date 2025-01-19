from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import time
import os

class ASR:
    def __init__(self, model_dir="models/SenseVoiceSmall", device="cuda:0"):
        print("初始化ASR模型...")
        start_time = time.time()
        self.model = AutoModel(
            model=model_dir,
            # vad_model="fsmn-vad",
            # vad_kwargs={"max_single_segment_time": 30000},
            device=device
        )
        end_time = time.time()
        print(f"ASR模型加载完成，耗时: {end_time - start_time:.2f}秒")

    def transcribe(self, audio_path):
        """
        将音频文件转换为文本
        Args:
            audio_path: 音频文件路径
        """
        try:
            res = self.model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            # 添加调试信息
            print(f"原始识别结果: {res[0]['text']}")
            
            text = rich_transcription_postprocess(res[0]["text"])
            
            # 如果结果包含大量非中文字符，可能是识别出现问题
            if len([c for c in text if '\u4e00' <= c <= '\u9fff']) < len(text) * 0.5:
                print("警告：识别结果包含大量非中文字符，可能存在识别错误")
            
            return text
        except Exception as e:
            print(f"转录过程出现错误: {str(e)}")
            return None

if __name__ == "__main__":
    # test
    asr = ASR()
    text = asr.transcribe("test.wav")
    print(text)

