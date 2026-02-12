import copy
import json
import warnings
import torch
import torchaudio
from transformers import AutoTokenizer
import os
import sys
import re
import yaml
import random
import numpy as np
from loguru import logger
from huggingface_hub import snapshot_download

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sentence_manager.sentence_manager import SentenceNormalizer
from spkemb_extractor import SpkembExtractor


def seed_everything(seed=1895):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
warnings.filterwarnings("ignore")


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


class MingAudio:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.eval().to(torch.bfloat16).to(self.device)

        if self.model.model_type == 'dense':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.sample_rate = self.model.config.audio_tokenizer_config.sample_rate
        self.patch_size = self.model.config.ditar_config['patch_size']
        self.normalizer = self.init_tn_normalizer(tokenizer=self.tokenizer)

        local_model_path = model_path if os.path.isdir(model_path) else snapshot_download(repo_id=model_path)
        self.spkemb_extractor = SpkembExtractor(f"{local_model_path}/campplus.onnx")


    def init_tn_normalizer(self, config_file_path=None, tokenizer=None):
        if config_file_path is None:
            default_config_path = "sentence_manager/default_config.yaml"
            config_file_path = default_config_path

        with open(config_file_path, 'r') as f:
            self.sentence_manager_config = yaml.safe_load(f)

        if "split_token" not in self.sentence_manager_config:
            self.sentence_manager_config["split_token"] = []

        assert isinstance(self.sentence_manager_config["split_token"], list)
        if tokenizer is not None:
            self.sentence_manager_config["split_token"].append(re.escape(tokenizer.eos_token))

        normalizer = SentenceNormalizer(self.sentence_manager_config.get("text_norm", {}))

        return normalizer

    def create_instruction(self, user_input: dict):
        new_caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
        target_item_dict = new_caption["audio_sequence"][0]

        for key, value in user_input.items():
            if key in target_item_dict:
                target_item_dict[key] = value

        if target_item_dict["BGM"].get("SNR", None) is not None:
            new_order = ["序号", "说话人", "BGM", "情感", "方言", "风格", "语速", "基频", "音量", "IP"]
            target_item_dict = {k: target_item_dict[k] for k in new_order if k in target_item_dict}
            new_caption["audio_sequence"][0] = target_item_dict

        return new_caption

    def pad_waveform(self, waveform):
        # Pad the prompt_waveform to ensure its length is a multiple of the patch size. 12.5 for tokenizer framerate.
        pad_align = int(1 / 12.5 * self.patch_size * self.sample_rate)
        new_len = (waveform.size(-1) + pad_align - 1) // pad_align * pad_align
        if new_len != waveform.size(1):
            new_wav = torch.zeros(1, new_len, dtype=waveform.dtype, device=waveform.device)
            new_wav[:, :waveform.size(1)] = waveform.clone()
            waveform = new_wav
        return waveform

    def preprocess_one_prompt_wav(self, waveform_path, use_spk_emb):
        if waveform_path is None:
            return None, None

        waveform, sr = torchaudio.load(waveform_path)
        waveform1 = waveform.clone()
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if use_spk_emb:
            waveform1 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform1)
            spk_emb = self.spkemb_extractor(waveform1)
        else:
            spk_emb = None
        return waveform, spk_emb

    def speech_generation(
        self,
        prompt,
        text,
        use_spk_emb=False,
        use_zero_spk_emb=False,
        instruction=None,
        prompt_wav_path=None,
        prompt_text=None,
        max_decode_steps=200,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        output_wav_path='./out.wav'
    ):
        # text = self.normalizer.normalize(text)
        if prompt_wav_path is None:
            prompt_waveform, prompt_text, spk_emb = None, None, None
            if use_zero_spk_emb:
                spk_emb = [torch.zeros(1, 192, device=self.device, dtype=torch.bfloat16)]
        else:
            paths = prompt_wav_path if isinstance(prompt_wav_path, list) else [prompt_wav_path]
            processed_prompts = [self.preprocess_one_prompt_wav(p, use_spk_emb) for p in paths]
            waveforms_list, spk_emb = zip(*processed_prompts)
            prompt_waveform = torch.cat(waveforms_list, dim=-1)
            prompt_waveform = self.pad_waveform(prompt_waveform)
            spk_emb = list(spk_emb)
            if all([x is None for x in spk_emb]):
                spk_emb = None

        if instruction is not None:
            instruction = self.create_instruction(instruction)
            instruction = json.dumps(instruction, ensure_ascii=False)

        waveform = self.model.generate(
            prompt=prompt,
            text=text,
            spk_emb=spk_emb,
            instruction=instruction,
            prompt_waveform=prompt_waveform,
            prompt_text=prompt_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            use_zero_spk_emb=use_zero_spk_emb
        )
        if output_wav_path is not None:
            output_dir = os.path.dirname(output_wav_path)
            os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_wav_path, waveform, sample_rate=self.sample_rate)
        return waveform

    def generation(
            self,
            prompt,
            text,
            max_decode_steps=200,
        ):
            text = self.model.generate_text(
                prompt=prompt,
                text=text,
                max_decode_steps=max_decode_steps,
            )
            return text


if __name__ == "__main__":
    model = MingAudio("inclusionAI/Ming-omni-tts-0.5B")
    # model = MingAudio("inclusionAI/Ming-omni-tta-0.5B")  # Only for TTA task
    # model = MingAudio("inclusionAI/Ming-omni-tts-16.8B-A3B")
    
    # Text Normalization
    # NOTE: The Text Normalization feature is not supported for MoE models.
    # decode_args = {
    #     "max_decode_steps": 200,
    # }
    # messages = {
    #     "prompt": "Please generate speech based on the following description.\n",
    #     "text": "化学反应方程式：\\ce{2H2 + O2 -> 2H2O}",
    # }

    # response = model.generation(**messages, **decode_args)
    # logger.info(f"Generated Response: {response}")

    # TTA
    decode_args = {
        "max_decode_steps": 200,
        "cfg": 4.5,
        "sigma": 0.3,
        "temperature": 2.5
    }
    messages = {
        "prompt": "Please generate audio events based on given text.\n",
        "text": "Thunder and a gentle rain",
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/tta.wav')
    logger.info(f"Generated Response: {response}")

    # Zero-shot TTS
    decode_args = {
        "max_decode_steps": 200,
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。",
        "use_spk_emb": True,
        "prompt_wav_path": "data/wavs/10002287-00000094.wav",
        "prompt_text": "在此奉劝大家别乱打美白针。"
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/tts.wav')
    logger.info(f"Generated Response: {response}")
    
    # BGM
    decode_args = {
        "max_decode_steps": 400,
    }
    attr = {
        "Genre": "电子舞曲.",
        "Mood": "自信 / 坚定.",
        "Instrument": "架子鼓.",
        "Theme": "节日.",
        "Duration": "30s."
    }
    text = " " + " ".join([f"{key}: {value}" for key, value in attr.items()])
    messages = {
        "prompt": "Please generate music based on the following description.\n",
        "text": text,
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/bgm.wav')
    logger.info(f"Generated Response: {response}")

    # Emotion
    decode_args = {
        "max_decode_steps": 200,
    }
    instruction = {
        "情感": "高兴"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我竟然抢到了陈奕迅的演唱会门票！太棒了！终于可以现场听一听他的歌声了！",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/emotion_prompt.wav",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/emotion.wav')
    logger.info(f"Generated Response: {response}")

    # Podcast
    decode_args = {
        "max_decode_steps": 200,
    }
    dialog = [
        {"speaker_1": "你可以说一下，就大概说一下，可能虽然我也不知道，我看过那部电影没有。"},
        {"speaker_2": "就是那个叫什么，变相一节课的嘛。"},
        {"speaker_1": "嗯。"},
        {"speaker_2": "一部搞笑的电影。"},
        {"speaker_1": "一部搞笑的。"}
    ]
    text = " " + "\n ".join([f"{k}:{v}" for item in dialog for k, v in item.items()]) + "\n"
    prompt_diag = [
        {"speaker_1": "并且我们还要进行每个月还要考核 笔试的话还要进行笔试，做个，当服务员还要去笔试了"},
        {"speaker_2": "对啊，这真的很奇怪，就是 单纯的因，单纯自己工资不高，只是因为可能人家那个店比较出名一点，就对你苛刻要求"},
    ]
    prompt_text = " " + "\n ".join([f"{k}:{v}" for item in prompt_diag for k, v in item.items()]) + "\n"

    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": text,
        "use_spk_emb": True,
        "prompt_wav_path": [
            "data/wavs/CTS-CN-F2F-2019-11-11-423-012-A.wav",
            "data/wavs/CTS-CN-F2F-2019-11-11-423-012-B.wav"
        ],
        "prompt_text": prompt_text
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/podcast.wav')
    logger.info(f"Generated Response: {response}")

    # Basic
    decode_args = {
        "max_decode_steps": 200,
    }
    instruction = {
        "语速": "快速",
        "基频": "中",
        "音量": "中",
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "简单地说，这相当于惠普把消费领域市场拱手相让了。",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/10002287-00000095.wav",
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/basic.wav')
    logger.info(f"Generated Response: {response}")

    # Dialect
    decode_args = {
        "max_decode_steps": 200
    }
    instruction = {
        "方言": "广粤话"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我觉得社会企业同个人都有责任",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/yue_prompt.wav",
    }

    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/dialect.wav')
    logger.info(f"Generated Response: {response}")
    
    # Timbre Definition
    decode_args = {
        "max_decode_steps": 200,
    }
    instruction = {
        "风格": "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。这个女性使用轻柔的普通话进行耳语，声音气音成分重。音量极低，紧贴麦克风，语速极慢，旨在制造触发听者颅内快感的声学刺激。"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？",
        "instruction": instruction,
        "use_zero_spk_emb": True
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/style.wav')
    logger.info(f"Generated Response: {response}")

    # IP
    decode_args = {
        "max_decode_steps": 200,
    }
    instruction = {
        "IP": "灵小甄"
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "这款产品的名字，叫变态坑爹牛肉丸。",
        "instruction": instruction,
        "use_zero_spk_emb": True
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/ip.wav')
    logger.info(f"Generated Response: {response}")

    # Speech + bgm
    decode_args = {
        "max_decode_steps": 200,
    }
    instruction = {
        "BGM": {"Genre": "当代古典音乐.", "Mood": "温暖 / 友善.", "Instrument": "电吉他", "Theme": "节日.", "SNR": 10.0, "ENV": None}
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "此次业绩下滑原因，可归结为企业停止服务某些品牌，而带来的负面影响。",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/00000309-00000300.wav",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/speech_bgm.wav')
    logger.info(f"Generated Response: {response}")

    # Speech+sound
    decode_args = {
        "max_decode_steps": 200,
    }
    instruction = {
        "BGM": {"ENV": "Birds chirping", "SNR": 10.0, "Genre": None, "Mood": None, "Instrument": None, "Theme": None}
    }
    messages = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "此次业绩下滑原因，可归结为企业停止服务某些品牌，而带来的负面影响。",
        "use_spk_emb": True,
        "instruction": instruction,
        "prompt_wav_path": "data/wavs/00000309-00000300.wav",
    }
    response = model.speech_generation(**messages, **decode_args, output_wav_path='output/speech_sound.wav')
    logger.info(f"Generated Response: {response}")
