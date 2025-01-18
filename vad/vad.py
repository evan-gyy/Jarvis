import torch
from hubconf import silero_vad
import numpy as np
import pyaudio
import wave
import time
import os


class VAD:
    def __init__(self):
        self.SAMPLING_RATE = 16000
        self.CHUNK = 512 
        self.THRESHOLD = 0.8
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        # 加载VAD模型
        self.model, self.vad_iterator = self.load_model()
        # 初始化音频流
        self.audio_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=self.SAMPLING_RATE, input=True, frames_per_buffer=self.CHUNK)

    def load_model(self):
        print("加载VAD模型...")
        start_time = time.time()
        model, utils = silero_vad(onnx=True, force_onnx_cpu=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        vad_iterator = VADIterator(model)
        end_time = time.time()
        print(f"VAD模型加载完成，耗时: {end_time - start_time:.2f} 秒")
        return model, vad_iterator

    def save_wav(self, audio_chunks, start_time, end_time, status):
        print("保存音频文件...")
        # 将音频数据块合并为一个数组
        audio_data = np.concatenate([np.frombuffer(chunk, dtype=np.int16) for chunk in audio_chunks])
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 构建输出文件名
        wav_filename = f"{self.output_dir}/output_{int(start_time)}_{status}.wav"
        
        # 保存为WAV文件
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位采样
            wf.setframerate(self.SAMPLING_RATE)
            wf.writeframes(audio_data)
            
        print(f"音频保存为 {wav_filename}，时长从 {start_time:.2f} 秒到 {end_time:.2f} 秒，状态：{status}")

    def process_audio_chunk(self, data):
        """处理单个音频数据块"""
        np_array = np.frombuffer(data, dtype=np.int16).copy()
        audio_chunk = torch.from_numpy(np_array).float() / 32768.0  # 归一化到[-1, 1]范围
        
        with torch.no_grad():
            speech_prob = self.model(audio_chunk.view(1, -1), self.SAMPLING_RATE).item()
            print(f"语音活动检测概率: {speech_prob}")
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
                current_time = time.time()
                print("停止检测到语音，保存录音")
                self.save_wav(audio_chunks, speech_state['start_time'], current_time, 0)
                speech_state['start_time'] = None
                speech_state['last_end_time'] = None
                return True
        return False

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

        finally:
            self.cleanup(speech_state, audio_chunks)

    def cleanup(self, speech_state, audio_chunks):
        """清理资源"""
        if speech_state['start_time'] is not None:
            current_time = time.time()
            print("异常结束，保存当前录音")
            self.save_wav(audio_chunks, speech_state['start_time'], current_time, 0)

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.vad_iterator.reset_states()
        print("资源释放完成")


if __name__ == "__main__":
    vad = VAD()
    vad.run()