from openai import OpenAI
from typing import List, Dict
import tiktoken
import os
from dotenv import load_dotenv
import time
import sys
from typing import Generator
from .prompt_template import LIANLIAN

class LLM:
    def __init__(self, api_key: str = None, max_history_turns: int = 10, max_tokens: int = 4000):
        """
        初始化LLM类
        Args:
            api_key: DeepSeek API密钥
            max_history_turns: 最大历史对话轮数
            max_tokens: 最大token数量
        """
        # 加载环境变量
        load_dotenv()
        
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key is None:
                raise ValueError("需要在.env文件中设置DEEPSEEK_API_KEY或直接提供api_key参数")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.max_history_turns = max_history_turns
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []
        self.system_prompt = LIANLIAN
        
        # 初始化tokenizer
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 使用GPT tokenizer作为近似
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.encoding.encode(text))
    
    def trim_history(self):
        """
        修剪历史记录，确保不超过最大轮数和token限制
        """
        trim_start_time = time.time()
        
        # 如果超过最大轮数，从前面删除
        while len(self.history) > self.max_history_turns:
            self.history.pop(0)
        
        # 计算当前token数量
        total_tokens = sum(self.count_tokens(msg["content"]) for msg in self.history)
        
        # 如果超过最大token数，从前面删除对话
        while total_tokens > self.max_tokens and self.history:
            removed_msg = self.history.pop(0)
            total_tokens -= self.count_tokens(removed_msg["content"])
            
        trim_time = (time.time() - trim_start_time) * 1000
        print(f"\n历史记录修剪耗时: {trim_time:.2f}ms")
    
    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """
        与模型进行对话，使用流式输出
        Args:
            user_input: 用户输入的文本
        Returns:
            模型的完整回复文本
        """
        try:
            # 添加用户输入到历史记录
            self.history.append({"role": "user", "content": user_input})
            
            # 构建完整的消息列表
            messages = [{"role": "system", "content": self.system_prompt}] + self.history
            
            print(f"\n用户: {user_input}")
            print("助手: ", end="", flush=True)
            
            # 调用API并计时
            api_start_time = time.time()
            first_token_received = False
            full_response = ""
            
            # 使用流式输出
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True
            )
            
            sentence_chunk = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    # 计算首个token的延迟
                    if not first_token_received:
                        first_token_time = (time.time() - api_start_time) * 1000
                        first_token_received = True
                    
                    # 流式输出文本
                    token = chunk.choices[0].delta.content
                    sentence_chunk += token
                    if token in ['。', '！', '？', ' ']:
                        print(sentence_chunk, end="", flush=True)
                        yield sentence_chunk
                        sentence_chunk = ""

                    full_response += token
            
            print(f"\n首token耗时: {first_token_time:.2f}ms")

            # 添加助手回复到历史记录
            self.history.append({"role": "assistant", "content": full_response})
            
            # 在所有处理完成后，最后进行历史记录的修剪
            self.trim_history()
            
            return full_response
            
        except Exception as e:
            print(f"\n调用API时出错: {str(e)}")
            # 发生错误时，移除刚才添加的用户输入，保持历史记录的一致性
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            return f"抱歉，发生了错误: {str(e)}"
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []


# 测试代码
if __name__ == "__main__":
    llm = LLM()
    for response in llm.stream_chat("你好，请介绍一下你自己"):
        pass