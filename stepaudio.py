import os

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from utils import load_audio, load_optimus_ths_lib, speech_adjust, volumn_adjust


class StepAudio:
    def __init__(self, tokenizer_path: str, tts_path: str, llm_path: str, device_map: str = "auto", gpu_ids: list = None):
        """
        初始化StepAudio模型，支持多GPU配置
        
        Args:
            tokenizer_path: 分词器路径
            tts_path: TTS模型路径
            llm_path: 语言模型路径
            device_map: 设备映射策略，可选"auto"、"balanced"、"sequential"等
            gpu_ids: 指定使用的GPU ID列表，如[0,1]
        """
        # 设置CUDA可见设备（如果指定了gpu_ids）
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            print(f"使用GPU: {gpu_ids}")
        
        load_optimus_ths_lib(os.path.join(llm_path, 'lib'))
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True
        )
        self.encoder = StepAudioTokenizer(tokenizer_path)
        self.decoder = StepAudioTTS(tts_path, self.encoder)
        
        # 使用指定的device_map加载LLM模型
        print(f"使用device_map策略: {device_map}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # 获取模型实际分布情况
        if hasattr(self.llm, "hf_device_map"):
            print(f"模型分布情况: {self.llm.hf_device_map}")

    def get_gpu_memory_info(self):
        """获取GPU内存使用情况"""
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**3)  # GB
            mem_used = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            mem_free = mem_total - mem_used
            
            gpu_memory.append({
                "device": i,
                "name": props.name,
                "total": round(mem_total, 2),
                "used": round(mem_used, 2),
                "free": round(mem_free, 2)
            })
        return gpu_memory

    def __call__(
        self,
        messages: list,
        speaker_id: str,
        speed_ratio: float = 1.0,
        volumn_ratio: float = 1.0,
    ):
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        
        # 将输入放到正确的设备上
        if hasattr(self.llm, "device") and self.llm.device is not None:
            token_ids = token_ids.to(self.llm.device)
            
        outputs = self.llm.generate(
            token_ids, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1] : -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        output_audio, sr = self.decoder(output_text, speaker_id)
        if speed_ratio != 1.0:
            output_audio = speech_adjust(output_audio, sr, speed_ratio)
        if volumn_ratio != 1.0:
            output_audio = volumn_adjust(output_audio, volumn_ratio)
        return output_text, output_audio, sr

    def encode_audio(self, audio_path):
        audio_wav, sr = load_audio(audio_path)
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    def apply_chat_template(self, messages: list):
        text_with_audio = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio += f"<|BOT|>{role}\n{content}<|EOT|>"
            elif isinstance(content, dict):
                if "type" in content and content["type"] == "text":
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif "type" in content and content["type"] == "audio":
                    audio_tokens = self.encode_audio(content["audio"])
                    text_with_audio += f"<|BOT|>{role}\n{audio_tokens}<|EOT|>"
                elif "audio" in content:  # 处理没有type但有audio的情况
                    audio_tokens = self.encode_audio(content["audio"])
                    text_with_audio += f"<|BOT|>{role}\n{audio_tokens}<|EOT|>"
            elif content is None:
                text_with_audio += f"<|BOT|>{role}\n"
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        if not text_with_audio.endswith("<|BOT|>assistant\n"):
            text_with_audio += "<|BOT|>assistant\n"
        return text_with_audio


if __name__ == "__main__":
    model = StepAudio(
        tokenizer_path="/mnt/ys-shai-jfs/open-step1o-audio/step1o-audio-encoder",
        tts_path="/mnt/ys-shai-jfs/open-step1o-audio/step1o-audio-decoder",
        llm_path="/mnt/ys-shai-jfs/open-step1o-audio/step1o-audio-v18",
        device_map="auto",  # 使用自动设备映射
    )

    text, audio, sr = model(
        [{"role": "user", "content": "你好，我是你的朋友，我叫小明，你叫什么名字？"}],
        "Tingting",
    )
    torchaudio.save("output/output_e2e_tqta.wav", audio, sr)
    text, audio, sr = model(
        [
            {
                "role": "user",
                "content": {"type": "audio", "audio": "output/output_e2e_tqta.wav"},
            }
        ],
        "Tingting",
    )
    torchaudio.save("output/output_e2e_aqta.wav", audio, sr)
