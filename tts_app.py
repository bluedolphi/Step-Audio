import gradio as gr
import argparse
import torchaudio
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from datetime import datetime
import os
from fastapi import FastAPI, HTTPException, Depends, Query, Body, File, UploadFile, Form
from fastapi.security import APIKeyHeader
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from status_tracker import StatusTracker

# 添加Token认证配置
DEFAULT_TOKEN = "step_audio_778899"
AUTH_TOKEN = os.environ.get("STEP_AUDIO_TOKEN", DEFAULT_TOKEN)

# 使用环境变量或默认值设置目录
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
LOGS_DIR = os.environ.get("LOGS_DIR", "./logs")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DATA_DIR, "cache"))

# 确保目录存在
os.makedirs(os.path.join(DATA_DIR, "output", "common"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "output", "music"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "output", "clone"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "cache"), exist_ok=True)

# 创建状态跟踪器（如果app.py未启动则创建新的，否则尝试复用）
try:
    from app import status_tracker
    print("复用app.py中的状态跟踪器")
except ImportError:
    status_tracker = StatusTracker()
    print("创建新的状态跟踪器")

# 创建FastAPI应用
app = FastAPI(
    title="Step-Audio TTS API",
    description="Step-Audio 语音合成服务API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# API安全相关
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Pydantic模型用于API请求和响应
class TTSCommonRequest(BaseModel):
    text: str
    speaker: str = "Tingting"
    emotion: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[str] = None
    token: str

class TTSMusicRequest(BaseModel):
    text: str
    speaker: str = "Tingting"
    mode: str = "RAP"
    token: str

class TTSResponse(BaseModel):
    status: str
    file_path: str
    text: str
    speaker: str
    params: Dict[str, Any]

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

# 依赖函数，用于API认证
async def verify_token(api_key: str = Depends(api_key_header)):
    if api_key == AUTH_TOKEN:
        return True
    raise HTTPException(
        status_code=401,
        detail="无效的API密钥",
        headers={"WWW-Authenticate": "Bearer"},
    )

# Token认证验证函数
def validate_token(token):
    """验证用户输入的token是否有效"""
    if token == AUTH_TOKEN:
        return True
    return False

# 保存音频
def save_audio(audio_type, audio_data, sr):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(DATA_DIR, "output", audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchaudio.save(save_path, audio_data, sr)
    return save_path


# 普通语音合成（添加Token认证）
def tts_common(text, speaker, emotion, language, speed, token):
    # 验证token
    if not validate_token(token):
        gr.Warning("认证失败，请输入有效的访问令牌")
        return None
        
    text = (
        (f"({emotion})" if emotion else "")
        + (f"({language})" if language else "")
        + (f"({speed})" if speed else "")
        + text
    )
    output_audio, sr = tts_engine(text, speaker)
    audio_type = "common"
    common_path = save_audio(audio_type, output_audio, sr)
    return common_path


# RAP / 哼唱模式（添加Token认证）
def tts_music(text_input_rap, speaker, mode_input, token):
    # 验证token
    if not validate_token(token):
        gr.Warning("认证失败，请输入有效的访问令牌")
        return None
        
    text_input_rap = f"({mode_input})" + text_input_rap
    output_audio, sr = tts_engine(text_input_rap, speaker)
    audio_type = "music"
    music_path = save_audio(audio_type, output_audio, sr)
    return music_path


# 语音克隆（添加Token认证）
def tts_clone(text, wav_file, speaker_prompt, emotion, language, speed, token):
    # 验证token
    if not validate_token(token):
        gr.Warning("认证失败，请输入有效的访问令牌")
        return None
        
    clone_speaker = {
        "wav_path": wav_file,
        "speaker": "custom_voice",
        "prompt_text": speaker_prompt,
    }
    clone_text = (
        (f"({emotion})" if emotion else "")
        + (f"({language})" if language else "")
        + (f"({speed})" if speed else "")
        + text
    )
    output_audio, sr = tts_engine(clone_text, "", clone_speaker)
    audio_type = "clone"
    clone_path = save_audio(audio_type, output_audio, sr)
    return clone_path


# API路由 - TTS相关接口
@app.post("/api/tts/common", response_model=TTSResponse, tags=["TTS"])
async def api_tts_common(request: TTSCommonRequest, auth: bool = Depends(verify_token)):
    """TTS普通语音合成API，支持情感、语言和语速控制"""
    if not validate_token(request.token):
        raise HTTPException(status_code=401, detail="认证失败，请提供有效的访问令牌")
    
    text_processed = (
        (f"({request.emotion})" if request.emotion else "")
        + (f"({request.language})" if request.language else "")
        + (f"({request.speed})" if request.speed else "")
        + request.text
    )
    output_audio, sr = tts_engine(text_processed, request.speaker)
    audio_type = "common"
    file_path = save_audio(audio_type, output_audio, sr)
    
    return {
        "status": "success",
        "file_path": file_path,
        "text": request.text,
        "speaker": request.speaker,
        "params": {
            "emotion": request.emotion,
            "language": request.language,
            "speed": request.speed
        }
    }

@app.post("/api/tts/music", response_model=TTSResponse, tags=["TTS"])
async def api_tts_music(request: TTSMusicRequest, auth: bool = Depends(verify_token)):
    """TTS音乐/哼唱模式API，支持RAP和哼唱模式"""
    if not validate_token(request.token):
        raise HTTPException(status_code=401, detail="认证失败，请提供有效的访问令牌")
    
    text_processed = f"({request.mode})" + request.text
    output_audio, sr = tts_engine(text_processed, request.speaker)
    audio_type = "music"
    file_path = save_audio(audio_type, output_audio, sr)
    
    return {
        "status": "success",
        "file_path": file_path,
        "text": request.text,
        "speaker": request.speaker,
        "params": {
            "mode": request.mode
        }
    }

@app.post("/api/tts/clone", tags=["TTS"])
async def api_tts_clone(
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    prompt_text: str = Form(...),
    speaker: str = Form("custom_voice"),
    emotion: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    speed: Optional[str] = Form(None),
    token: str = Form(...),
    auth: bool = Depends(verify_token)
):
    """TTS语音克隆API，使用参考音频进行语音克隆"""
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="认证失败，请提供有效的访问令牌")
    
    # 保存上传的音频文件
    temp_audio_path = os.path.join(CACHE_DIR, f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    # 构建克隆参数
    clone_speaker = {
        "wav_path": temp_audio_path,
        "speaker": speaker,
        "prompt_text": prompt_text,
    }
    
    # 处理文本
    clone_text = (
        (f"({emotion})" if emotion else "")
        + (f"({language})" if language else "")
        + (f"({speed})" if speed else "")
        + text
    )
    
    # 生成音频
    output_audio, sr = tts_engine(clone_text, "", clone_speaker)
    audio_type = "clone"
    file_path = save_audio(audio_type, output_audio, sr)
    
    # 返回结果
    return {
        "status": "success",
        "file_path": file_path,
        "text": text,
        "speaker": "cloned",
        "params": {
            "reference_audio": audio_file.filename,
            "prompt_text": prompt_text,
            "emotion": emotion,
            "language": language,
            "speed": speed
        }
    }

@app.get("/api/tts/audio/{file_path:path}", tags=["TTS"])
async def get_audio_file(file_path: str, auth: bool = Depends(verify_token)):
    """获取生成的音频文件"""
    full_path = os.path.join(DATA_DIR, "output", file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    
    return FileResponse(full_path, media_type="audio/wav")


def _launch_demo(args, tts_engine):
    # 选项列表
    emotion_options = ["高兴1", "高兴2", "生气1", "生气2", "悲伤1", "撒娇1"]
    language_options = ["中文", "英文", "韩语", "日语", "四川话", "粤语", "广东话"]
    speed_options = ["慢速1", "慢速2", "快速1", "快速2"]
    speaker_options = ["Tingting"]
    # Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("## 🎙️ Step-Audio-TTS-3B Demo")

        # 添加Token认证输入框
        auth_token = gr.Textbox(
            label="Authentication Token",
            type="password",
            placeholder="请输入访问令牌",
        )

        # 普通语音合成
        with gr.Tab("Common TTS (普通语音合成)"):
            text_input = gr.Textbox(
                label="Input Text (输入文本)",
            )
            speaker_input = gr.Dropdown(
                speaker_options,
                label="Speaker Selection (音色选择)",
            )
            emotion_input = gr.Dropdown(
                emotion_options,
                label="Emotion Style (情感风格)",
                allow_custom_value=True,
                interactive=True,
            )
            language_input = gr.Dropdown(
                language_options,
                label="Language/Dialect (语言/方言)",
                allow_custom_value=True,
                interactive=True,
            )
            speed_input = gr.Dropdown(
                speed_options,
                label="Speech Rate (语速调节)",
                allow_custom_value=True,
                interactive=True,
            )
            submit_btn = gr.Button("🔊 Generate Speech (生成语音)")
            output_audio = gr.Audio(
                label="Output Audio (合成语音)",
                interactive=False,
            )

            submit_btn.click(
                tts_common,
                inputs=[
                    text_input,
                    speaker_input,
                    emotion_input,
                    language_input,
                    speed_input,
                    auth_token,
                ],
                outputs=output_audio,
            )

        # RAP / 哼唱模式
        with gr.Tab("RAP/Humming Mode (RAP/哼唱模式)"):
            text_input_rap = gr.Textbox(
                label="Lyrics Input (歌词输入)",
            )
            speaker_input = gr.Dropdown(
                speaker_options,
                label="Speaker Selection (音色选择)",
            )
            mode_input = gr.Radio(
                ["RAP", "Humming (哼唱)"],
                value="RAP",
                label="Generation Mode (生成模式)",
            )
            submit_btn_rap = gr.Button("🎤 Generate Performance (生成演绎)")
            output_audio_rap = gr.Audio(
                label="Performance Audio (演绎音频)", interactive=False
            )
            submit_btn_rap.click(
                tts_music,
                inputs=[
                    text_input_rap, 
                    speaker_input, 
                    mode_input,
                    auth_token,
                ],
                outputs=output_audio_rap,
            )

        with gr.Tab("Voice Clone (语音克隆)"):
            text_input_clone = gr.Textbox(
                label="Target Text (目标文本)",
                placeholder="Text to be synthesized with cloned voice (待克隆语音合成的文本)",
            )
            audio_input = gr.File(
                label="Reference Audio Upload (参考音频上传)",
            )
            speaker_prompt = gr.Textbox(
                label="Exact text from reference audio (输入参考音频的准确文本)",
            )
            emotion_input = gr.Dropdown(
                emotion_options,
                label="Emotion Style (情感风格)",
                allow_custom_value=True,
                interactive=True,
            )
            language_input = gr.Dropdown(
                language_options,
                label="Language/Dialect (语言/方言)",
                allow_custom_value=True,
                interactive=True,
            )
            speed_input = gr.Dropdown(
                speed_options,
                label="Speech Rate (语速调节)",
                allow_custom_value=True,
                interactive=True,
            )
            submit_btn_clone = gr.Button("🗣️ Synthesize Cloned Speech (合成克隆语音)")
            output_audio_clone = gr.Audio(
                label="Cloned Speech Output (克隆语音输出)",
                interactive=False,
            )
            submit_btn_clone.click(
                tts_clone,
                inputs=[
                    text_input_clone,
                    audio_input,
                    speaker_prompt,
                    emotion_input,
                    language_input,
                    speed_input,
                    auth_token,
                ],
                outputs=output_audio_clone,
            )

    # 将Gradio应用挂载到FastAPI应用
    gr_app = gr.mount_gradio_app(app, demo, path="/tts")
    return app


if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    import uvicorn

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Demo server port."
    )
    parser.add_argument("--tmp_dir", type=str, default="/tmp/gradio", help="Save path.")

    args = parser.parse_args()
    # 使用解析后的命令行参数设置模型路径
    model_path = args.model_path
    
    # 加载模型
    tokenizer_path = os.path.join(model_path, "Step-Audio-Tokenizer")
    tts_model_path = os.path.join(model_path, "Step-Audio-TTS-3B")
    
    encoder = StepAudioTokenizer(tokenizer_path)
    tts_engine = StepAudioTTS(tts_model_path, encoder)
    
    # 更新模型状态
    status_tracker.update_model_status("tokenizer", True, tokenizer_path)
    status_tracker.update_model_status("tts", True, tts_model_path)
    
    # 启动应用
    app = _launch_demo(args, tts_engine)
    
    # 启动服务器
    uvicorn.run(
        app, 
        host=args.server_name, 
        port=args.server_port
    )
