from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
from typing import Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager
from funasr import AutoModel
from funasr_onnx import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import tempfile
import wave

class ASRServer:
    def __init__(self, model_dir="models/SenseVoiceSmall", device="cuda:0", use_onnx=False):
        print("初始化ASR模型...")
        start_time = time.time()
        
        if use_onnx:
            self.model = SenseVoiceSmall(
                model_dir, 
                batch_size=1,
                quantize=True
            )
        else:
            self.model = AutoModel(
                model=model_dir,
                trust_remote_code=True,
                device=device
            )
            
        end_time = time.time()
        print(f"ASR模型加载完成，耗时: {end_time - start_time:.2f}秒")
        self.use_onnx = use_onnx

    def transcribe(self, audio_data: bytes, batch_size: int = 64) -> Optional[str]:
        """
        转写音频数据
        Args:
            audio_data: WAV音频数据
            batch_size: 批处理大小
        """
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            try:
                # 开始计时
                transcribe_start = time.time()
                
                if self.use_onnx:
                    res = self.model(
                        [temp_path],
                        language="zh",
                        use_itn=True
                    )
                    text = rich_transcription_postprocess(res[0])
                else:
                    res = self.model.generate(
                        input=temp_path,
                        cache={},
                        language="zh",
                        use_itn=True,
                        batch_size=batch_size,
                    )
                    text = rich_transcription_postprocess(res[0]["text"])
                
                # 计算转录耗时
                transcribe_time = (time.time() - transcribe_start) * 1000
                print(f"转录耗时: {transcribe_time:.2f}ms")
                
                return text
                
            finally:
                # 确保删除临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            print(f"转录过程出现错误: {str(e)}")
            return None

# 全局ASR实例
asr_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理应用的生命周期事件"""
    global asr_instance
    try:
        asr_instance = ASRServer(use_onnx=False)
        yield
    except Exception as e:
        print(f"ASR模型加载失败: {str(e)}")
        raise e
    finally:
        pass

# 创建FastAPI应用
app = FastAPI(
    title="ASR Service",
    description="语音识别服务API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscribeResponse(BaseModel):
    """转写响应模型"""
    text: str
    elapsed_time: float
    success: bool
    message: Optional[str] = None

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    batch_size: int = 64
):
    """转写音频文件"""
    try:
        # 读取上传的音频数据
        audio_data = await file.read()
        
        # 开始计时
        start_time = time.time()
        
        # 调用ASR进行识别
        text = asr_instance.transcribe(audio_data, batch_size=batch_size)
        
        # 计算耗时
        elapsed_time = (time.time() - start_time) * 1000
        
        if text:
            return TranscribeResponse(
                text=text,
                elapsed_time=elapsed_time,
                success=True
            )
        else:
            return TranscribeResponse(
                text="",
                elapsed_time=elapsed_time,
                success=False,
                message="识别失败"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": asr_instance is not None}

def start_server(host="0.0.0.0", port=8000):
    """启动服务器"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server(port=49180) 