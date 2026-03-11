import os
import json
import random
import copy
import re
import yaml
import uuid
from pathlib import Path

import torch
import torchaudio
import gradio as gr
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from configuration_bailingmm import BailingMMConfig
from sentence_manager.sentence_manager import SentenceNormalizer
from spkemb_extractor import SpkembExtractor

# All model IDs from cookbook / test.py
MODEL_CHOICES = [
    "inclusionAI/Ming-omni-tts-0.5B",        # Dense — TTS + TTA + BGM
    "inclusionAI/Ming-omni-tts-16.8B-A3B",    # MoE   — TTS + TTA + BGM (no TN)
    "inclusionAI/Ming-omni-tta-0.5B",          # Dense — TTA only
]
DEFAULT_MODEL = MODEL_CHOICES[0]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_CACHE = {"model": None, "key": None}

BASE_CAPTION_TEMPLATE = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}

# Per-task recommended decode defaults matching cookbook/test.py values
TASK_DEFAULTS = {
    "TTS (Speech)":       {"max_decode_steps": 200, "cfg": 2.0, "sigma": 0.25, "temperature": 0.0},
    "TTA (Audio Events)": {"max_decode_steps": 200, "cfg": 4.5, "sigma": 0.3,  "temperature": 2.5},
    "Music (BGM)":        {"max_decode_steps": 400, "cfg": 2.0, "sigma": 0.25, "temperature": 0.0},
}


def resolve_llm_dtype(device):
    if device.startswith("cuda"):
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float32
        )
    return torch.float32


def set_attn_backend(config_like, backend: str):
    if config_like is None:
        return
    if isinstance(config_like, dict):
        config_like["attn_implementation"] = backend
        config_like["_attn_implementation"] = backend
        return
    config_like.attn_implementation = backend
    config_like._attn_implementation = backend


def resolve_qwen_attention_backend(device) -> str:
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        return "eager"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def seed_everything(seed=1895):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MingRuntime:
    def __init__(self, model_path, device):
        self.device = device
        self.llm_dtype = resolve_llm_dtype(device)

        if not os.path.isdir(model_path):
            local_model_path = snapshot_download(model_path)
        else:
            local_model_path = model_path

        config = BailingMMConfig.from_pretrained(local_model_path)
        set_attn_backend(config, "eager")

        nested_attn_impl = resolve_qwen_attention_backend(device)
        self.nested_attn_impl = nested_attn_impl
        set_attn_backend(config.llm_config, nested_attn_impl)

        if hasattr(config, "audio_tokenizer_config"):
            if hasattr(config.audio_tokenizer_config, "enc_kwargs"):
                if "backbone" in config.audio_tokenizer_config.enc_kwargs:
                    set_attn_backend(config.audio_tokenizer_config.enc_kwargs["backbone"], nested_attn_impl)
            if hasattr(config.audio_tokenizer_config, "dec_kwargs"):
                if "backbone" in config.audio_tokenizer_config.dec_kwargs:
                    set_attn_backend(config.audio_tokenizer_config.dec_kwargs["backbone"], nested_attn_impl)

        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=self.llm_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.eval().to(self.llm_dtype).to(device)
        # Non-bf16 CUDA GPUs such as T4 run directly in float32.

        if self.model.model_type == "dense":
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)

        self.model.tokenizer = self.tokenizer
        self.sample_rate = self.model.config.audio_tokenizer_config.sample_rate
        self.patch_size = self.model.config.ditar_config["patch_size"]
        self.model_type = self.model.model_type

        self.spk = SpkembExtractor(f"{local_model_path}/campplus.onnx")
        self.normalizer = self.init_tn_normalizer()

    def init_tn_normalizer(self, config_file_path=None):
        if config_file_path is None:
            config_file_path = "sentence_manager/default_config.yaml"
        try:
            with open(config_file_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load TN config: {e}")
            config = {}
        if "split_token" not in config:
            config["split_token"] = []
        assert isinstance(config["split_token"], list)
        config["split_token"].append(re.escape(self.tokenizer.eos_token))
        return SentenceNormalizer(config.get("text_norm", {}))

    def create_instruction(self, user_input: dict):
        new_caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
        target_item_dict = new_caption["audio_sequence"][0]
        for key, value in user_input.items():
            if key == "BGM" and isinstance(value, dict):
                target_item_dict["BGM"].update(value)
            elif key in target_item_dict:
                target_item_dict[key] = value
        if target_item_dict["BGM"].get("SNR", None) is not None:
            new_order = ["序号", "说话人", "BGM", "情感", "方言", "风格", "语速", "基频", "音量", "IP"]
            target_item_dict = {k: target_item_dict[k] for k in new_order if k in target_item_dict}
            new_caption["audio_sequence"][0] = target_item_dict
        return new_caption

    def pad_waveform(self, waveform):
        pad_align = int(1 / 12.5 * self.patch_size * self.sample_rate)
        new_len = (waveform.size(-1) + pad_align - 1) // pad_align * pad_align
        if new_len != waveform.size(1):
            new_wav = torch.zeros(1, new_len, dtype=waveform.dtype, device=waveform.device)
            new_wav[:, :waveform.size(1)] = waveform.clone()
            waveform = new_wav
        return waveform

    def preprocess_one_prompt_wav(self, waveform_path, use_spk_emb):
        if not waveform_path:
            return None, None
        waveform, sr = torchaudio.load(waveform_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        waveform1 = waveform.clone()
        if use_spk_emb:
            waveform1 = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)(waveform1)
            spk_emb = self.spk(waveform1).to(self.device, dtype=torch.float32)
        else:
            spk_emb = None
        return waveform, spk_emb

    def generate_audio(
        self,
        task_type,
        text,
        ref_audios=None,
        prompt_text=None,
        use_tn=False,
        use_spk_emb=True,
        use_zero_spk_emb=False,
        instruction_dict=None,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
        max_decode_steps=200,
    ):
        if use_tn:
            text = self.normalizer.normalize(text)

        if task_type == "TTA (Audio Events)":
            prompt = "Please generate audio events based on given text.\n"
        elif task_type == "Music (BGM)":
            prompt = "Please generate music based on the following description.\n"
        else:
            prompt = "Please generate speech based on the following description.\n"

        if not ref_audios:
            prompt_waveform = None
            spk_emb = None
            if use_zero_spk_emb:
                # float32 is safe on T4; cookbook uses bfloat16 but that requires A100/H100
                spk_emb = [torch.zeros(1, 192, device=self.device, dtype=torch.float32)]
        else:
            paths = ref_audios if isinstance(ref_audios, list) else [ref_audios]
            processed_prompts = [self.preprocess_one_prompt_wav(p, use_spk_emb) for p in paths if p]
            waveforms_list = [x[0] for x in processed_prompts if x[0] is not None]
            spk_embs = [x[1] for x in processed_prompts]
            prompt_waveform = None
            if waveforms_list:
                prompt_waveform = torch.cat(waveforms_list, dim=-1)
                prompt_waveform = self.pad_waveform(prompt_waveform).to(self.device, dtype=torch.float32)
            spk_emb = [x for x in spk_embs if x is not None]
            if not spk_emb:
                spk_emb = None

        final_instruction = None
        if instruction_dict:
            final_instruction = self.create_instruction(instruction_dict)
            final_instruction = json.dumps(final_instruction, ensure_ascii=False)

        waveform = self.model.generate(
            prompt=prompt,
            text=text,
            spk_emb=spk_emb,
            instruction=final_instruction,
            prompt_waveform=prompt_waveform,
            prompt_text=prompt_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            use_zero_spk_emb=use_zero_spk_emb,
        )

        waveform = waveform.float()
        has_nan = torch.isnan(waveform).any().item()
        has_inf = torch.isinf(waveform).any().item()
        print(f"waveform dtype={waveform.dtype} shape={tuple(waveform.shape)} nan={has_nan} inf={has_inf}")
        if has_nan or has_inf:
            print("WARNING: NaN/Inf in waveform — replacing with zeros. "
                  "Verify audio decoder sub-modules were cast to float32.")
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            print(f"  min={waveform.min().item():.4f}  max={waveform.max().item():.4f}  "
                  f"abs_mean={waveform.abs().mean().item():.4f}")

        out_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
        torchaudio.save(str(out_path), waveform.cpu(), self.sample_rate)
        return str(out_path)

    def generate_text(self, text, max_decode_steps=200):
        """
        Text-only generation via model.generate_text().
        NOTE: Not supported on MoE models.
        """
        prompt = "Please generate speech based on the following description.\n"
        return self.model.generate_text(
            prompt=prompt,
            text=text,
            max_decode_steps=max_decode_steps,
        )


def load_model(model_path, device):
    key = f"{model_path}::{device}::{resolve_llm_dtype(device)}"
    if MODEL_CACHE["model"] and MODEL_CACHE["key"] == key:
        return runtime_info(MODEL_CACHE["model"])
    runtime = MingRuntime(model_path, device)
    MODEL_CACHE["model"] = runtime
    MODEL_CACHE["key"] = key
    return runtime_info(runtime)


def unload_model():
    MODEL_CACHE["model"] = None
    MODEL_CACHE["key"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model unloaded."


def runtime_info(runtime):
    return json.dumps(
        {
            "device": runtime.device,
            "dtype": str(runtime.llm_dtype),
            "nested_attention_backend": runtime.nested_attn_impl,
            "sample_rate": runtime.sample_rate,
            "model_type": runtime.model_type,
            "tn_supported": runtime.model_type == "dense",
            "text_generation_supported": runtime.model_type == "dense",
        },
        indent=2,
    )


def update_task_defaults(task_type):
    """Auto-fill decode sliders with per-task cookbook defaults."""
    d = TASK_DEFAULTS.get(task_type, TASK_DEFAULTS["TTS (Speech)"])
    return d["max_decode_steps"], d["cfg"], d["sigma"], d["temperature"]


def build_podcast_text(dialog_json_str):
    """
    Convert a JSON dialog list into the speaker_N:... format the model expects.
    e.g. [{"speaker_1":"Hello"},{"speaker_2":"World"}]
      -> ' speaker_1:Hello\n speaker_2:World\n'
    """
    try:
        dialog = json.loads(dialog_json_str)
        lines = [f"{k}:{v}" for item in dialog for k, v in item.items()]
        return " " + "\n ".join(lines) + "\n"
    except Exception as e:
        return f"[JSON parse error: {e}]"


def generate_audio_wrapper(
    task_type, text, podcast_mode, podcast_json,
    ref_audios, prompt_text, use_tn,
    use_spk_emb, use_zero_spk_emb,
    ip, emotion, dialect, style, speed, pitch, volume,
    bgm_genre, bgm_mood, bgm_instrument, bgm_theme, bgm_env, bgm_snr, bgm_duration,
    cfg, sigma, temperature, max_decode_steps,
):
    runtime = MODEL_CACHE["model"]
    if runtime is None:
        raise gr.Error("Please load the model first.")

    # Podcast mode: build structured dialog text
    if task_type == "TTS (Speech)" and podcast_mode:
        if not podcast_json.strip():
            raise gr.Error("Podcast mode is enabled but the dialog JSON is empty.")
        built = build_podcast_text(podcast_json)
        if built.startswith("[JSON parse error"):
            raise gr.Error(f"Invalid podcast dialog JSON: {built}")
        text = built

    # TN check
    if use_tn and runtime.model_type != "dense":
        raise gr.Error("Text Normalization is not supported for MoE models.")

    # Build instruction dict
    instruction = {}
    if emotion: instruction["情感"] = emotion
    if dialect: instruction["方言"] = dialect
    if style:   instruction["风格"] = style
    if speed:   instruction["语速"] = speed
    if pitch:   instruction["基频"] = pitch
    if volume:  instruction["音量"] = volume
    if ip:      instruction["IP"] = ip

    bgm_dict = {}
    if bgm_genre:      bgm_dict["Genre"] = bgm_genre
    if bgm_mood:       bgm_dict["Mood"] = bgm_mood
    if bgm_instrument: bgm_dict["Instrument"] = bgm_instrument
    if bgm_theme:      bgm_dict["Theme"] = bgm_theme
    if bgm_env:        bgm_dict["ENV"] = bgm_env
    if bgm_snr not in (None, 0, 0.0): bgm_dict["SNR"] = bgm_snr
    if bgm_dict:
        instruction["BGM"] = bgm_dict
    instruction = instruction if instruction else None

    # TTA: clear incompatible inputs
    if task_type == "TTA (Audio Events)":
        instruction = None
        use_spk_emb = False
        use_zero_spk_emb = False
        prompt_text = ""
        ref_audios = None

    # Music: compile BGM attrs into generation text
    if task_type == "Music (BGM)":
        music_attrs = []
        if bgm_genre:      music_attrs.append(f"Genre: {bgm_genre}.")
        if bgm_mood:       music_attrs.append(f"Mood: {bgm_mood}.")
        if bgm_instrument: music_attrs.append(f"Instrument: {bgm_instrument}.")
        if bgm_theme:      music_attrs.append(f"Theme: {bgm_theme}.")
        if bgm_duration:   music_attrs.append(f"Duration: {bgm_duration}.")
        if bgm_env:        music_attrs.append(f"ENV: {bgm_env}.")
        if bgm_snr not in (None, 0, 0.0): music_attrs.append(f"SNR: {bgm_snr}.")
        if text.strip():   music_attrs.append(f"Text: {text}")
        text = " " + " ".join(music_attrs)

    # Validation
    has_music_condition = any([
        text.strip(), bgm_genre, bgm_mood, bgm_instrument,
        bgm_theme, bgm_env, bgm_duration, bgm_snr not in (None, 0, 0.0),
    ])
    if task_type == "Music (BGM)" and not has_music_condition:
        raise gr.Error("For Music, provide text or at least one BGM attribute.")
    elif task_type != "Music (BGM)" and not text.strip():
        raise gr.Error("Text input cannot be empty.")

    ref_audio_paths = [f.name if hasattr(f, "name") else f for f in ref_audios] if ref_audios else None

    return runtime.generate_audio(
        task_type=task_type,
        text=text,
        ref_audios=ref_audio_paths,
        prompt_text=prompt_text if prompt_text.strip() else None,
        use_tn=use_tn,
        use_spk_emb=use_spk_emb,
        use_zero_spk_emb=use_zero_spk_emb,
        instruction_dict=instruction,
        cfg=cfg,
        sigma=sigma,
        temperature=temperature,
        max_decode_steps=int(max_decode_steps),
    )


def generate_text_wrapper(text, max_decode_steps):
    runtime = MODEL_CACHE["model"]
    if runtime is None:
        raise gr.Error("Please load the model first.")
    if not text.strip():
        raise gr.Error("Text input cannot be empty.")
    if runtime.model_type != "dense":
        raise gr.Error("Text Generation (generate_text) is only supported on dense models.")
    return str(runtime.generate_text(text=text, max_decode_steps=int(max_decode_steps)))


def build_ui():
    with gr.Blocks(title="Ming Omni Audio Demo") as demo:
        gr.Markdown("# Ming Omni Audio Demo")
        gr.Markdown(
            "Supported Tasks: **TTS** · **TTA** · **Music/BGM** · **Zero-shot Clone** · "
            "**Podcast (multi-speaker)** · **Emotion / Dialect / Style / IP / BGM mix** · "
            "**Text Generation** · **All 3 model variants**"
        )

        # Model loader 
        with gr.Row():
            model_path = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                allow_custom_value=True,
                label="Model (Hub ID or local path)",
                scale=3,
                info="Ming-omni-tta-0.5B supports TTA only. MoE 16.8B does not support TN or Text Generation.",
            )
            device = gr.Dropdown(["cuda:0"], value="cuda:0", label="Device", scale=1)
        with gr.Row():
            load_btn   = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload Model")
        runtime_box = gr.Code(label="Runtime Info (shows model_type, tn_supported, etc.)", language="json", lines=6)

        # Tabs
        with gr.Tabs():

            # TAB 1: Audio Generation 
            with gr.TabItem("Audio Generation"):
                with gr.Row():
                    with gr.Column(scale=2):

                        task_type = gr.Radio(
                            ["TTS (Speech)", "TTA (Audio Events)", "Music (BGM)"],
                            label="Task Type",
                            value="TTS (Speech)",
                        )

                        text = gr.Textbox(
                            label="Target Text",
                            placeholder=(
                                "TTS: plain text\n"
                                "TTA: describe the sound event (e.g. 'Thunder and a gentle rain')\n"
                                "Music: describe style/mood, or use BGM fields below\n"
                                "Podcast: leave blank and use the Podcast Builder accordion"
                            ),
                            lines=4,
                        )

                        use_tn = gr.Checkbox(
                            label="Enable Text Normalization (TN) — dense models only, not MoE",
                            value=False,
                        )

                        # Podcast / Multi-speaker Builder 
                        with gr.Accordion("🎙️ Podcast / Multi-Speaker Builder", open=False):
                            gr.Markdown(
                                "Enable to build a structured multi-speaker dialog. "
                                "Upload **one reference audio per speaker** "
                                "(speaker_1 first, speaker_2 second, etc.) in the Reference Audio section. "
                                "When enabled, the Target Text above is ignored and replaced by the dialog here."
                            )
                            podcast_mode = gr.Checkbox(label="Enable Podcast Mode", value=False)
                            podcast_json = gr.Textbox(
                                label='Dialog JSON — list of {"speaker_N": "line"} objects',
                                placeholder=(
                                    '[\n'
                                    '  {"speaker_1": "你好，今天天气真好。"},\n'
                                    '  {"speaker_2": "是啊，适合出去走走。"},\n'
                                    '  {"speaker_1": "那我们去公园吧！"}\n'
                                    ']'
                                ),
                                lines=7,
                            )
                            podcast_preview = gr.Textbox(
                                label="Compiled text preview (what gets sent to model)",
                                interactive=False,
                                lines=3,
                            )
                            podcast_json.change(
                                fn=lambda j: build_podcast_text(j) if j.strip() else "",
                                inputs=podcast_json,
                                outputs=podcast_preview,
                            )

                        # Reference Audio
                        with gr.Accordion("🎤 Reference Audio (Zero-shot / Podcast)", open=True):
                            ref_audios = gr.File(
                                label="Reference Audio(s) — upload multiple for Podcast/multi-speaker",
                                file_count="multiple",
                                file_types=["audio"],
                            )
                            prompt_text = gr.Textbox(label="Reference Transcript", lines=2)
                            with gr.Row():
                                use_spk_emb = gr.Checkbox(
                                    label="Use Speaker Embedding Extraction",
                                    value=True,
                                )
                                use_zero_spk_emb = gr.Checkbox(
                                    label="Use Zero Speaker Embedding (for IP / Style)",
                                    value=False,
                                )

                        # Voice Attributes
                        with gr.Accordion("🗣️ Voice Attributes & IP", open=False):
                            ip      = gr.Textbox(label="IP Name (e.g., 灵小甄)")
                            emotion = gr.Textbox(label="Emotion (e.g., 高兴)")
                            dialect = gr.Textbox(label="Dialect (e.g., 广粤话)")
                            style   = gr.Textbox(label="Style Definition (e.g., ASMR耳语...)")
                            with gr.Row():
                                speed  = gr.Textbox(label="Speed (e.g., 快速)",  placeholder="Optional")
                                pitch  = gr.Textbox(label="Pitch (e.g., 中)",    placeholder="Optional")
                                volume = gr.Textbox(label="Volume (e.g., 高)",   placeholder="Optional")

                        # BGM 
                        with gr.Accordion("🎵 BGM & Environmental Sounds", open=False):
                            gr.Markdown(
                                "**TTS + BGM mix**: set SNR + traits (requires reference audio).  \n"
                                "**Music generation**: these fields become the generation prompt.  \n"
                                "**Speech + Env sound**: set ENV + SNR only."
                            )
                            with gr.Row():
                                bgm_genre      = gr.Textbox(label="Genre (e.g., 当代古典音乐.)")
                                bgm_mood       = gr.Textbox(label="Mood (e.g., 温暖 / 友善.)")
                                bgm_instrument = gr.Textbox(label="Instrument (e.g., 电吉他)")
                            with gr.Row():
                                bgm_theme = gr.Textbox(label="Theme (e.g., 节日.)")
                                bgm_env   = gr.Textbox(label="ENV (e.g., Birds chirping)")
                                bgm_snr   = gr.Number(label="SNR dB (TTS+BGM mix, 0 = off)", value=0.0)
                            with gr.Row():
                                bgm_duration = gr.Textbox(label="Duration (e.g., 30s.)", placeholder="Music only")

                        # Decode Settings
                        with gr.Accordion("⚙️ Decode Settings", open=False):
                            gr.Markdown(
                                "Defaults auto-update when Task Type changes to match cookbook values  \n"
                                "(TTA: cfg=4.5 / temp=2.5 / sigma=0.3 — Music: steps=400)"
                            )
                            max_decode_steps = gr.Slider(50, 1000, value=200, step=10,  label="max_decode_steps")
                            cfg              = gr.Slider(0.0, 10.0, value=2.0, step=0.1,   label="cfg (Guidance Scale)")
                            sigma            = gr.Slider(0.0, 1.0,  value=0.25, step=0.01, label="sigma")
                            temperature      = gr.Slider(0.0, 5.0,  value=0.0, step=0.1,   label="temperature")

                        # Auto-update decode sliders on task change
                        task_type.change(
                            fn=update_task_defaults,
                            inputs=task_type,
                            outputs=[max_decode_steps, cfg, sigma, temperature],
                        )

                    with gr.Column(scale=1):
                        generate_btn = gr.Button("▶ Generate Audio", variant="primary", size="lg")
                        out_audio    = gr.Audio(label="Generated Audio", interactive=False)

                generate_btn.click(
                    generate_audio_wrapper,
                    inputs=[
                        task_type, text, podcast_mode, podcast_json,
                        ref_audios, prompt_text, use_tn,
                        use_spk_emb, use_zero_spk_emb,
                        ip, emotion, dialect, style, speed, pitch, volume,
                        bgm_genre, bgm_mood, bgm_instrument, bgm_theme, bgm_env, bgm_snr, bgm_duration,
                        cfg, sigma, temperature, max_decode_steps,
                    ],
                    outputs=out_audio,
                )

            # TAB 2: Text Generation 
            with gr.TabItem("Text Generation (generate_text)"):
                gr.Markdown(
                    "Direct text generation via `model.generate_text()` — "
                    "equivalent to the **Text Normalization** cookbook cell.  \n"
                    "**Only supported on dense models** (Ming-omni-tts-0.5B), not MoE (16.8B-A3B)."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        tg_text  = gr.Textbox(
                            label="Input Text",
                            placeholder="化学反应方程式：\\ce{2H2 + O2 -> 2H2O}",
                            lines=4,
                        )
                        tg_steps = gr.Slider(50, 500, value=200, step=10, label="max_decode_steps")
                        tg_btn   = gr.Button("▶ Generate Text", variant="primary")
                    with gr.Column(scale=1):
                        tg_out = gr.Textbox(label="Model Text Output", lines=6, interactive=False)

                tg_btn.click(
                    generate_text_wrapper,
                    inputs=[tg_text, tg_steps],
                    outputs=tg_out,
                )

        # ── Model load/unload ──────────────────────────────────────────────────
        load_btn.click(load_model,    inputs=[model_path, device], outputs=runtime_box)
        unload_btn.click(unload_model, outputs=runtime_box)

    return demo


if __name__ == "__main__":
    seed_everything()
    app = build_ui()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
