import sounddevice as sd
import soundfile as sf
import threading
from queue import Queue, Empty
import time
import os
import glob
from traceback import print_exc

class AudioPlayer:
    def __init__(self, vad=None):
        self.vad = vad
        self.audio_queue = Queue()
        self.is_playing = False
        self.max_temp_files = 10
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        self.first_audio_start_time = None
        self.running = True
        self.stream_processor = None
        self.should_resume_recording = False  # 添加标志来控制录音恢复
        
        # 启动播放线程
        self.player_thread = threading.Thread(target=self._player_worker)
        self.player_thread.daemon = False  # 改为非守护线程
        self.player_thread.start()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker)
        self.cleanup_thread.daemon = True  # 清理线程可以保持为守护线程
        self.cleanup_thread.start()
    
    def _player_worker(self):
        """音频播放线程"""
        while self.running:
            try:
                audio_file = self.audio_queue.get(timeout=0.1)
                if audio_file:
                    self.play_audio(audio_file)
            except Empty:
                continue
            except Exception as e:
                print_exc()
                print(f"播放音频时出错: {e}")
                continue
    
    def _cleanup_worker(self):
        """清理临时文件的线程"""
        while True:
            try:
                # 当播放队列为空时进行清理
                if self.audio_queue.empty() and not self.is_playing:
                    self.cleanup_temp_files()
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                print_exc()
                print(f"清理临时文件时出错: {e}")
    
    def cleanup_temp_files(self):
        """清理临时文件，只保留最新的N个文件"""
        try:
            # 获取所有wav文件
            wav_files = glob.glob(os.path.join(self.temp_dir, "*.wav"))
            
            # 如果文件数量超过限制
            if len(wav_files) > self.max_temp_files:
                # 按修改时间排序
                wav_files.sort(key=lambda x: os.path.getmtime(x))
                
                # 删除最旧的文件
                files_to_delete = wav_files[:-self.max_temp_files]
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        # print(f"已删除旧音频文件: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {str(e)}")
                        
        except Exception as e:
            print_exc()
            print(f"清理WAV文件时出错: {str(e)}")
    
    def set_stream_processor(self, processor):
        """设置StreamProcessor引用"""
        self.stream_processor = processor

    def play_audio(self, file_path):
        """播放单个音频文件"""
        try:
            # 暂停录音
            if self.vad and not self.is_playing:  # 只在第一次播放时暂停
                self.vad.pause_recording()
                self.is_playing = True
            
            # 记录第一个音频开始播放的时间
            if self.first_audio_start_time is None:
                self.first_audio_start_time = time.time()
                if hasattr(self.vad, 'pipeline_start_time'):
                    latency = self.first_audio_start_time - self.vad.pipeline_start_time
                    print(f"首次响应延迟: {latency:.2f}s")
            
            # 播放音频
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
            
            # 更新已播放句子计数
            if self.stream_processor:
                self.stream_processor.played_sentences += 1
                # 检查是否是最后一个句子
                if self.stream_processor.played_sentences >= self.stream_processor.total_sentences:
                    self.should_resume_recording = True
                    self.is_playing = False
                    if self.vad:
                        self.vad.resume_recording()
                
        except Exception as e:
            print_exc()
            print(f"播放音频失败: {e}")
            if self.vad:
                self.is_playing = False
                self.vad.resume_recording()
    
    def add_to_queue(self, audio_file):
        """添加音频文件到播放队列"""
        self.audio_queue.put(audio_file)
        # print(list(self.audio_queue.queue))
    
    def is_queue_empty(self):
        """检查播放队列是否为空"""
        return self.audio_queue.empty()
    
    def reset_timing(self):
        """重置计时器和状态"""
        self.first_audio_start_time = None
        self.should_resume_recording = False
    
    def stop(self):
        """停止播放器"""
        self.running = False
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break 