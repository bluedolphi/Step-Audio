import gradio as gr
import time
from pathlib import Path
import torchaudio
from stepaudio import StepAudio
import os
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.security import APIKeyHeader
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from status_tracker import StatusTracker

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 添加Token认证配置
DEFAULT_TOKEN = "step_audio_778899"
AUTH_TOKEN = os.environ.get("STEP_AUDIO_TOKEN", DEFAULT_TOKEN)

# 使用环境变量或默认值设置目录
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
LOGS_DIR = os.environ.get("LOGS_DIR", "./logs")

# 确保目录存在
os.makedirs(os.path.join(DATA_DIR, "output"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "cache"), exist_ok=True)

# 调整临时缓存目录
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# 创建状态跟踪器
status_tracker = StatusTracker()

# 创建FastAPI应用
app = FastAPI(
    title="Step-Audio API",
    description="Step-Audio 语音合成和语音聊天服务API",
    version="1.0.0",
    docs_url=None,  # 禁用默认的文档URL
    redoc_url=None, # 禁用默认的ReDoc URL
)

# API安全相关
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Pydantic模型用于API请求和响应
class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: int

class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total: float
    memory_used: float
    memory_free: float
    utilization: float

class GPUResourceResponse(BaseModel):
    gpu_count: int
    gpus: List[GPUInfo]

class ModelInfo(BaseModel):
    loaded: bool
    path: Optional[str] = None
    load_time: Optional[str] = None

class ModelStatusResponse(BaseModel):
    models: Dict[str, ModelInfo]
    ready: bool

class CPUInfo(BaseModel):
    percent: float
    cores: int

class MemoryInfo(BaseModel):
    total: float
    available: float
    percent: float

class DiskInfo(BaseModel):
    total: float
    free: float
    percent: float

class SystemResourceResponse(BaseModel):
    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo

class MessageRequest(BaseModel):
    text: str
    token: str

class MessageResponse(BaseModel):
    response: str
    audio_path: str


# 依赖函数，用于API认证
async def verify_token(api_key: str = Depends(api_key_header)):
    if api_key == AUTH_TOKEN:
        return True
    raise HTTPException(
        status_code=401,
        detail="无效的API密钥",
        headers={"WWW-Authenticate": "Bearer"},
    )


class CustomAsr:
    def __init__(self, model_name="iic/SenseVoiceSmall", device="cuda"):
        self.model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
        )

    def run(self, audio_path):
        res = self.model.generate(
            input=audio_path,
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text


def add_message(chatbot, history, mic, text):
    if not mic and not text:
        return chatbot, history, "Input is empty"

    if text:
        chatbot.append({"role": "user", "content": text})
        history.append({"role": "user", "content": text})
    elif mic and Path(mic).exists():
        chatbot.append({"role": "user", "content": {"path": mic}})
        history.append({"role": "user", "content": {"type":"audio", "audio": mic}})

    print(f"{history=}")
    return chatbot, history, None


def reset_state(system_prompt):
    """Reset the chat history."""
    return [], [{"role": "system", "content": system_prompt}]


def save_tmp_audio(audio, sr):
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=CACHE_DIR, delete=False, suffix=".wav"
    ) as temp_audio:
        temp_audio_path = temp_audio.name
        torchaudio.save(temp_audio_path, audio, sr)

    return temp_audio.name


def predict(chatbot, history, audio_model, asr_model):
    """Generate a response from the model."""
    try:
        is_input_audio = False
        user_audio_path = None
        # 检测用户输入的是音频还是文本
        if isinstance(history[-1]["content"], dict):
            is_input_audio = True
            user_audio_path = history[-1]["content"]["audio"]
        text, audio, sr = audio_model(history, "Tingting")
        print(f"predict {text=}")
        audio_path = save_tmp_audio(audio, sr)
        # 缓存用户语音的 asr 文本结果为了加速下一次推理
        if is_input_audio:
            asr_text = asr_model.run(user_audio_path)
            chatbot.append({"role": "user", "content": asr_text})
            history[-1]["content"] = asr_text
            print(f"{asr_text=}")
        chatbot.append({"role": "assistant", "content": {"path": audio_path}})
        chatbot.append({"role": "assistant", "content": text})
        history.append({"role": "assistant", "content": text})
    except Exception as e:
        print(e)
        gr.Warning(f"Some error happend, retry submit")
    return chatbot, history


# Token认证验证函数
def validate_token(token):
    """验证用户输入的token是否有效"""
    if token == AUTH_TOKEN:
        return True
    return False


# 自定义文档路由
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI页面路由"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """自定义ReDoc页面路由"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI schema端点"""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )


# API路由
@app.get("/api/health", response_model=HealthResponse, tags=["状态"])
async def health_check():
    """健康检查接口，返回服务状态信息"""
    return status_tracker.get_health_info()

@app.get("/api/model/status", response_model=ModelStatusResponse, tags=["状态"])
async def model_status(auth: bool = Depends(verify_token)):
    """模型状态检查接口，返回模型加载状态"""
    return status_tracker.get_model_status()

@app.get("/api/resources/gpu", response_model=GPUResourceResponse, tags=["资源"])
async def gpu_resources(auth: bool = Depends(verify_token)):
    """GPU资源检查接口，返回GPU使用情况"""
    return status_tracker.get_gpu_info()

@app.get("/api/resources/system", response_model=SystemResourceResponse, tags=["资源"])
async def system_resources(auth: bool = Depends(verify_token)):
    """系统资源检查接口，返回CPU、内存和磁盘使用情况"""
    return status_tracker.get_system_info()

@app.post("/api/chat/message", response_model=MessageResponse, tags=["聊天"])
async def chat_message(
    request: MessageRequest = Body(..., description="聊天请求"),
    auth: bool = Depends(verify_token)
):
    """发送聊天消息并获取回复，支持文本输入，返回文本和语音回复"""
    global audio_model, asr_model
    
    if not validate_token(request.token):
        raise HTTPException(
            status_code=401,
            detail="认证失败，请提供有效的访问令牌"
        )
    
    # 构建历史记录
    history = [
        {"role": "system", "content": "适配用户的语言，用简短口语化的文字回答"},
        {"role": "user", "content": request.text}
    ]
    
    # 生成回复
    text, audio, sr = audio_model(history, "Tingting")
    audio_path = save_tmp_audio(audio, sr)
    
    return {
        "response": text,
        "audio_path": audio_path
    }


def _launch_demo(args, audio_model, asr_model):
    with gr.Blocks(delete_cache=(86400, 86400)) as demo:
        gr.Markdown("""<center><font size=8>Step Audio Chat</center>""")
        
        # 添加Token认证输入框
        auth_token = gr.Textbox(
            label="Authentication Token",
            type="password",
            placeholder="请输入访问令牌",
        )
        
        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="适配用户的语言，用简短口语化的文字回答",
                lines=2
            )
            
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            avatar_images=["assets/user.png", "assets/assistant.png"],
            min_height=800,
            type="messages",
        )
        # 保存 chat 历史，不需要每次再重新拼格式
        history = gr.State([{"role": "system", "content": system_prompt.value}])
        mic = gr.Audio(type="filepath")
        text = gr.Textbox(placeholder="Enter message ...")

        with gr.Row():
            clean_btn = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            submit_btn = gr.Button("🚀 Submit")

        def on_submit(chatbot, history, mic, text, token):
            # 先验证token
            if not validate_token(token):
                gr.Warning("认证失败，请输入有效的访问令牌")
                return chatbot, history, mic, text
                
            chatbot, history, error = add_message(
                chatbot, history, mic, text
            )
            if error:
                gr.Warning(error)  # 显示警告消息
                return chatbot, history, None, None
            else:
                chatbot, history = predict(chatbot, history, audio_model, asr_model)
                return chatbot, history, None, None

        submit_btn.click(
            fn=on_submit,
            inputs=[chatbot, history, mic, text, auth_token],
            outputs=[chatbot, history, mic, text],
            concurrency_limit=4,
            concurrency_id="gpu_queue",
        )
        
        def reset_with_auth(system_prompt, token):
            # 验证token
            if not validate_token(token):
                gr.Warning("认证失败，请输入有效的访问令牌")
                return chatbot, history
            return reset_state(system_prompt)
            
        clean_btn.click(
            fn=reset_with_auth,
            inputs=[system_prompt, auth_token],
            outputs=[chatbot, history],
            show_progress=True,
        )

        def regenerate_with_auth(chatbot, history, token):
            # 验证token
            if not validate_token(token):
                gr.Warning("认证失败，请输入有效的访问令牌")
                return chatbot, history
                
            while chatbot and chatbot[-1]["role"] == "assistant":
                chatbot.pop()
            while history and history[-1]["role"] == "assistant":
                print(f"discard {history[-1]}")
                history.pop()
            return predict(chatbot, history, audio_model, asr_model)

        regen_btn.click(
            regenerate_with_auth,
            [chatbot, history, auth_token],
            [chatbot, history],
            show_progress=True,
            concurrency_id="gpu_queue",
        )

    # 将Gradio应用挂载到FastAPI应用
    gr_app = gr.mount_gradio_app(app, demo, path="/")
    return app


if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    import uvicorn

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable sharing of the demo."
    )
    parser.add_argument(
        "--gpu-ids", type=str, default="", help="GPU IDs to use, comma separated (e.g. '0,1')"
    )
    parser.add_argument(
        "--device-map", type=str, default="auto", 
        help="Model loading device mapping strategy (auto, balanced, sequential)"
    )
    args = parser.parse_args()

    # 解析GPU配置
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None
    
    # 设置模型路径
    tokenizer_path = os.path.join(args.model_path, "Step-Audio-Tokenizer")
    tts_path = os.path.join(args.model_path, "Step-Audio-TTS-3B")
    llm_path = os.path.join(args.model_path, "Step-Audio-Chat")
    
    # 初始化模型
    audio_model = StepAudio(
        tokenizer_path=tokenizer_path,
        tts_path=tts_path,
        llm_path=llm_path,
        device_map=args.device_map,
        gpu_ids=gpu_ids
    )
    
    # 更新模型状态
    status_tracker.update_model_status("tokenizer", True, tokenizer_path)
    status_tracker.update_model_status("tts", True, tts_path)
    status_tracker.update_model_status("llm", True, llm_path)
    
    asr_model = CustomAsr()
    app = _launch_demo(args, audio_model, asr_model)
    
    # 启动服务器
    uvicorn.run(
        app, 
        host=args.server_name, 
        port=args.server_port
    )