import time
import os
from statistics import mean
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr.asr import ASR
from llm.llm import LLM
from tts.tts import TTS

class PerformanceTest:
    def __init__(self, test_file='test.wav', mode='full'):
        """
        初始化性能测试类
        Args:
            test_file: 测试音频文件名
            mode: 计算模式，'full' 或 'first'
                 'full': 计算完整响应的时间
                 'first': 只计算第一个句子的时间
        """
        # 初始化各个组件
        self.asr = ASR()
        self.llm = LLM()
        self.tts = TTS()
        self.mode = mode
        
        # 存储测试结果
        self.metrics = {
            'asr_time': [],
            'llm_first_token': [],
            'tts_first_audio': [],
            'total_pipeline': []
        }
        
        # 测试音频路径
        self.test_audio = os.path.join(os.path.dirname(__file__), test_file)
        
        # 添加结果保存路径
        self.excel_path = os.path.join(os.path.dirname(__file__), 'performance_results.xlsx')

    def single_test(self):
        """执行单次测试"""
        results = {}
        
        # ASR 测试
        asr_start = time.time()
        text = self.asr.transcribe(self.test_audio)
        asr_end = time.time()
        results['asr_time'] = asr_end - asr_start
        
        if not text:
            print("ASR 识别失败")
            return None
            
        # LLM 测试 - 每次重新初始化 LLM
        self.llm = LLM()  # 重新初始化 LLM 对象
        llm_start = time.time()
        first_token_time = None
        full_response = ""
        first_sentence = ""
        
        for response_chunk in self.llm.stream_chat(text):
            if response_chunk == "__END__":
                break
                
            if first_token_time is None:
                first_token_time = time.time() - llm_start
                
            full_response += response_chunk
            
            # 如果是 first 模式且找到了第一个完整句子
            if self.mode == 'first' and not first_sentence and any(p in response_chunk for p in ['。', '！', '？']):
                first_sentence = full_response
                break  # first 模式下找到第一个句子就退出
                
        results['llm_first_token'] = first_token_time
        
        # TTS 测试
        tts_start = time.time()
        first_audio_time = None
        
        # 根据模式选择要合成的文本
        text_to_synthesize = first_sentence if self.mode == 'first' and first_sentence else full_response
        
        audio_file = self.tts.synthesize(text_to_synthesize)
        if audio_file:
            first_audio_time = time.time() - tts_start
        
        results['tts_first_audio'] = first_audio_time
        
        # 计算总流程时间
        results['total_pipeline'] = time.time() - asr_start
        
        return results

    def run_tests(self, n=5):
        """运行多次测试并计算统计数据"""
        print(f"\n开始执行 {n} 次性能测试...\n")
        
        for i in range(n):
            print(f"执行第 {i+1}/{n} 次测试...")
            results = self.single_test()
            
            if results:
                for metric, value in results.items():
                    self.metrics[metric].append(value)
            
        self.print_statistics()

    def print_statistics(self):
        """打印统计结果"""
        print("\n性能测试统计结果:")
        print(f"计算模式: {'完整响应' if self.mode == 'full' else '首句响应'}")
        print("-" * 85)
        
        # 表头
        headers = ["指标", "平均耗时(ms)", "标准差(ms)", "最小值(ms)", "最大值(ms)"]
        header_format = "{:<15} {:<15} {:<15} {:<15} {:<15}"
        print(header_format.format(*headers))
        print("-" * 85)
        
        # 按固定顺序显示指标
        metrics_order = [
            ('asr_time', 'ASR 识别'),
            ('llm_first_token', 'LLM 首Token'),
            ('tts_first_audio', 'TTS 首音频'),
            ('total_pipeline', '总流程')
        ]
        
        # 数据行
        row_format = "{:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}"
        for metric_key, metric_name in metrics_order:
            values = self.metrics[metric_key]
            if values:  # 只显示有数据的指标
                mean_value = mean(values) * 1000
                std_value = np.std(values) * 1000
                min_value = min(values) * 1000
                max_value = max(values) * 1000
                print(row_format.format(
                    metric_name,
                    mean_value,
                    std_value,
                    min_value,
                    max_value
                ))
        
        print("-" * 85)
        print("注: 所有时间均从 ASR 开始计时")

    def get_mean_metrics(self):
        """获取所有指标的平均值"""
        return {
            metric: mean(values) * 1000 if values else 0  # 转换为毫秒
            for metric, values in self.metrics.items()
        }
        
    def save_to_excel(self, results_dict=None):
        """
        保存结果到Excel
        Args:
            results_dict: 包含其他测试结果的字典，格式为 {mode: {metric: value}}
        """
        # 获取当前测试的平均值
        current_results = self.get_mean_metrics()
        
        if results_dict is None:
            results_dict = {}
        results_dict[self.mode] = current_results
        
        # 创建DataFrame
        df = pd.DataFrame(results_dict).T
        
        # 重新排列列的顺序
        columns_order = [
            'asr_time',
            'llm_first_token',
            'tts_first_audio',
            'total_pipeline'
        ]
        
        # 重命名列
        column_names = {
            'asr_time': 'ASR耗时(ms)',
            'llm_first_token': 'LLM首Token(ms)',
            'tts_first_audio': 'TTS首音频(ms)',
            'total_pipeline': '总流程(ms)'
        }
        
        # 重新排序并重命名列
        df = df[columns_order].rename(columns=column_names)
        
        # 重命名索引
        mode_names = {
            'full': '完整响应',
            'first': '首句响应'
        }
        df.index = df.index.map(mode_names)
        
        # 保存到Excel
        df.to_excel(self.excel_path)
        print(f"\n结果已保存到: {self.excel_path}")
        
        return results_dict

if __name__ == "__main__":
    results_dict = {}
    
    # 完整响应模式测试
    print("\n=== 完整响应模式测试 ===")
    tester_full = PerformanceTest(test_file='test.wav', mode='full')
    tester_full.run_tests(5)
    results_dict = tester_full.save_to_excel(results_dict)
    
    # 首句响应模式测试
    print("\n=== 首句响应模式测试 ===")
    tester_first = PerformanceTest(test_file='test.wav', mode='first')
    tester_first.run_tests(5)
    tester_first.save_to_excel(results_dict) 