import torch
from .hubconf import silero_vad
import numpy as np
import pyaudio
import wave
import time
import os
import glob
from traceback import print_exc
from asr.asr import ASR
from llm.llm import LLM
from tts.tts import TTS
from .workflow import StreamProcessor
import asyncio


class VAD:
    def __init__(self):
        self.SAMPLING_RATE = 16000
        self.CHUNK = 512 
        self.THRESHOLD = 0.8
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        self.max_wav_files = 10  # 最大保留的WAV文件数量
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        # 初始清理
        self.cleanup_wav_files()
        
        # 添加上次打印时间的记录
        self.last_prob_print_time = 0
        self.print_interval = 0.5  # 打印间隔设为0.5秒，即每秒打印2次
        # 加载VAD模型
        self.model, self.vad_iterator = self.load_model()
        self.is_recording_paused = False  # 添加标志来追踪录音状态
        self.audio = pyaudio.PyAudio()
        self.init_audio_stream()
        # 初始化ASR模型
        self.asr = ASR()
        # 初始化LLM
        self.llm = LLM()
        # 初始化TTS
        self.tts = TTS()
        self.tts.set_vad(self)  # 设置 VAD 实例到 TTS
        self.is_playing_audio = False  # 添加标志来追踪是否正在播放音频
        self.stream_processor = StreamProcessor(self.llm, self.tts)
        self.tts.audio_player.set_stream_processor(self.stream_processor)  # 设置引用

    def load_model(self):
        """加载Silero VAD模型"""
        print("加载VAD模型...")
        start_time = time.time()
        model, utils = silero_vad(onnx=True, force_onnx_cpu=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        vad_iterator = VADIterator(model)
        end_time = time.time()
        print(f"VAD模型加载完成，耗时: {end_time - start_time:.2f} 秒")
        return model, vad_iterator

    def cleanup_wav_files(self):
        """
        清理WAV文件，只保留最新的N个文件
        """
        try:
            # 获取所有wav文件
            wav_files = glob.glob(os.path.join(self.output_dir, "*.wav"))
            
            # 如果文件数量超过限制
            if len(wav_files) > self.max_wav_files:
                # 按修改时间排序
                wav_files.sort(key=lambda x: os.path.getmtime(x))
                
                # 计算需要删除的文件数量
                files_to_delete = len(wav_files) - self.max_wav_files
                
                # 删除最旧的文件
                for i in range(files_to_delete):
                    try:
                        os.remove(wav_files[i])
                        print(f"已删除旧音频文件: {os.path.basename(wav_files[i])}")
                    except Exception as e:
                        print(f"删除文件 {wav_files[i]} 时出错: {str(e)}")
                
        except Exception as e:
            print(f"清理WAV文件时出错: {str(e)}")

    def save_wav(self, audio_chunks, start_time, end_time, status):
        """保存WAV文件并进行语音识别"""
        print("保存音频文件...")
        
        # 保存逻辑
        audio_data = np.concatenate([np.frombuffer(chunk, dtype=np.int16) for chunk in audio_chunks])
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 构建输出文件名
        wav_filename = f"{self.output_dir}/output_{int(start_time)}_{status}.wav"
        
        # 保存为WAV文件
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLING_RATE)
            wf.writeframes(audio_data)
            
        print(f"音频保存为 {wav_filename}，时长为 {(end_time - start_time):.2f} s，状态：{status}")
        
        # 保存完音频后立即进行ASR识别
        text = None
        if os.path.exists(wav_filename):
            print("开始进行语音识别...")
            start_time = time.time()
            text = self.asr.transcribe(wav_filename)
            end_time = time.time()
            print(f"语音识别完成，耗时: {(end_time - start_time) * 1000:.2f} ms")
            if text:
                print(f"识别结果: {text}")
        
        # 在语音识别完成后进行文件清理
        self.cleanup_wav_files()
            
        return text

    def process_audio_chunk(self, data):
        """处理单个音频数据块"""
        # 如果正在播放音频，跳过录音处理
        if self.tts.audio_player.is_playing:
            return 0.0
            
        np_array = np.frombuffer(data, dtype=np.int16).copy()
        audio_chunk = torch.from_numpy(np_array).float() / 32768.0
        
        with torch.no_grad():
            speech_prob = self.model(audio_chunk.view(1, -1), self.SAMPLING_RATE).item()
            
            current_time = time.time()
            if current_time - self.last_prob_print_time >= self.print_interval:
                print(f"语音检测概率: {speech_prob:.2f}")
                self.last_prob_print_time = current_time
                
            return speech_prob

    def handle_speech_detection(self, speech_prob, speech_state):
        """处理语音检测状态"""
        if speech_prob > self.THRESHOLD:
            if speech_state['start_time'] is None:
                speech_state['start_time'] = time.time()
                if speech_state['first_start_time'] == 0:
                    speech_state['first_start_time'] = speech_state['start_time']
                    print("first_start_time:", speech_state['first_start_time'])
                print("开始检测到语音")
            speech_state['last_end_time'] = None
            return True
        return False

    def handle_silence(self, speech_state, audio_chunks):
        """处理静音状态"""
        if speech_state['start_time'] is not None:
            if speech_state['last_end_time'] is None:
                speech_state['last_end_time'] = time.time()
            
            if time.time() - speech_state['last_end_time'] >= 1.0:
                try:
                    current_time = time.time()
                    print("停止检测到语音，保存录音并进行识别")
                    
                    # 开始计时
                    self.pipeline_start_time = time.time()
                    # 重置音频播放器的计时
                    self.tts.audio_player.reset_timing()
                    
                    # 语音识别
                    text = self.save_wav(audio_chunks, speech_state['start_time'], current_time, 0)
                    speech_state['start_time'] = None
                    speech_state['last_end_time'] = None
                    
                    # 使用新的流式处理
                    if text:
                        asyncio.run(self.stream_processor.process_stream(text))
                        self.stream_processor.wait_for_completion()
                        
                    return True
                
                except Exception as e:
                    print(f"处理静音状态时出错: {str(e)}")
                    # 重置状态
                    speech_state['start_time'] = None
                    speech_state['last_end_time'] = None
                    return True  # 返回True以清空音频块
                
        return False

    def init_audio_stream(self):
        """初始化音频流"""
        try:
            if hasattr(self, 'audio_stream'):
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            # 重新初始化 PyAudio 实例
            if hasattr(self, 'audio'):
                self.audio.terminate()
            self.audio = pyaudio.PyAudio()
            
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.SAMPLING_RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except Exception as e:
            print_exc()
            print(f"初始化音频流时出错: {e}")
            # 如果出错，等待一段时间后重试
            time.sleep(0.5)
            try:
                self.audio = pyaudio.PyAudio()
                self.audio_stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.SAMPLING_RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )
            except Exception as e:
                print_exc()
                print(f"重试初始化音频流时出错: {e}")
                raise

    def pause_recording(self):
        """暂停录音"""
        if not self.is_recording_paused and hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.is_recording_paused = True
            except Exception as e:
                print_exc()
                print(f"暂停录音时出错: {e}")

    def resume_recording(self):
        """恢复录音"""
        if self.is_recording_paused:
            try:
                if hasattr(self, 'audio_stream') and self.audio_stream:
                    self.audio_stream.start_stream()
                else:
                    self.init_audio_stream()
                self.is_recording_paused = False
            except Exception as e:
                print_exc()
                print(f"恢复录音时出错: {e}")
                # 如果启动失败，尝试重新初始化
                try:
                    self.init_audio_stream()
                    self.is_recording_paused = False
                except Exception as e:
                    print_exc()
                    print(f"重新初始化音频流时出错: {e}")

    def run(self):
        """主运行函数"""
        try:
            speech_state = {
                'start_time': None,
                'last_end_time': None,
                'first_start_time': 0
            }
            audio_chunks = []

            while True:
                data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                speech_prob = self.process_audio_chunk(data)
                
                is_speech = self.handle_speech_detection(speech_prob, speech_state)
                if is_speech:
                    audio_chunks.append(data)
                else:
                    if self.handle_silence(speech_state, audio_chunks):
                        audio_chunks = []

        except KeyboardInterrupt:
            print("\n录音结束")

        except Exception as e:
            print_exc()
            print(f"发生错误: {e}")

        finally:
            self.cleanup(speech_state, audio_chunks)

    def cleanup(self, speech_state, audio_chunks):
        """清理资源"""
        try:
            # 先停止音频播放器
            if hasattr(self, 'tts') and hasattr(self.tts, 'audio_player'):
                self.tts.audio_player.stop()

            # 停止流处理器
            if hasattr(self, 'stream_processor'):
                self.stream_processor.stop()
                self.stream_processor.wait_for_completion()

            if speech_state['start_time'] is not None:
                current_time = time.time()
                print("异常结束，保存当前录音")
                self.save_wav(audio_chunks, speech_state['start_time'], current_time, 0)

            if hasattr(self, 'audio_stream'):
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            if hasattr(self, 'audio'):
                self.audio.terminate()
                
            self.vad_iterator.reset_states()
            print("资源释放完成")
        except Exception as e:
            print_exc()
            print(f"清理资源时出错: {str(e)}")


if __name__ == "__main__":
    vad = VAD()
    vad.run()