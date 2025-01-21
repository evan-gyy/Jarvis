import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from queue import Queue, Empty
import threading
import time
from llm.llm import LLM
from tts.tts import TTS
from traceback import print_exc

class StreamProcessor:
    def __init__(self, llm, tts):
        self.llm = llm
        self.tts = tts
        self.sentence_queue = Queue()
        self.running = True
        self.total_sentences = 0  # 总句子数
        self.played_sentences = 0  # 已播放句子数
        
        # 启动TTS消费线程
        self.tts_thread = threading.Thread(target=self._tts_consumer)
        self.tts_thread.daemon = False
        self.tts_thread.start()

    def _tts_consumer(self):
        """TTS消费线程"""
        while self.running:
            try:
                sentence = self.sentence_queue.get(timeout=0.1)
                if sentence:
                    # 合成音频并获取文件路径
                    audio_file = self.tts.synthesize(sentence)
                    if audio_file:
                        # 将音频文件添加到播放队列
                        self.tts.audio_player.add_to_queue(audio_file)
            except Empty:
                continue
            except Exception as e:
                print_exc()
                print(f"TTS消费者线程出错: {str(e)}")
                continue

    async def process_stream(self, text):
        """处理流式输出"""
        try:
            self.total_sentences = 0  # 重置计数
            self.played_sentences = 0
            # 调用LLM获取流式响应
            for sentence in self.llm.stream_chat(text):
                if not self.running:
                    break
                if sentence == "__END__":  # 不处理结束标记
                    continue
                self.sentence_queue.put(sentence)
                self.total_sentences += 1
        except Exception as e:
            print_exc()
            print(f"处理流式输出时出错: {str(e)}")

    def wait_for_completion(self):
        """等待所有音频播放完成"""
        try:
            while self.played_sentences < self.total_sentences:
                time.sleep(0.1)
        except Exception as e:
            print_exc()
            print(f"等待完成时出错: {str(e)}")

    def stop(self):
        """停止处理器"""
        self.running = False
        # 清空队列
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except Empty:
                break


if __name__ == "__main__":
    llm = LLM()
    tts = TTS()
    stream_processor = StreamProcessor(llm, tts)
    asyncio.run(stream_processor.process_stream("你好，请介绍一下你自己"))
