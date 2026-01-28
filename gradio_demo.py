import gc
import inspect
import logging
import re
from argparse import ArgumentParser
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from mmaudio.utils.cache_utils import configure_local_cache

configure_local_cache()

import gradio as gr
import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, VideoInfo, all_model_cfg, generate, load_image,
                                load_video, make_video, setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    log.warning('CUDA/MPS are not available, running on CPU')
dtype = torch.bfloat16

model: ModelConfig = all_model_cfg['large_44k_v2']
model.download_if_needed()
output_dir = Path('./output/gradio')
TEXT_TO_AUDIO_SAMPLES = 5

setup_eval_logging()


def get_interface_kwargs() -> dict:
    params = inspect.signature(gr.Interface).parameters
    kwargs: dict = {}
    if 'submit_btn' in params:
        kwargs['submit_btn'] = '生成'
    if 'clear_btn' in params:
        kwargs['clear_btn'] = '清空'
    return kwargs


interface_kwargs = get_interface_kwargs()

def make_audio_output(label: str = '输出音频') -> gr.Audio:
    params = inspect.signature(gr.Audio).parameters
    kwargs = {'label': label}
    if 'show_download_button' in params:
        kwargs['show_download_button'] = True
    return gr.Audio(**kwargs)

def sanitize_prompt_prefix(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r'\W+', '_', text.strip(), flags=re.UNICODE).strip('_')
    if not cleaned:
        cleaned = 'prompt'
    return cleaned[:max_len]


def get_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = model.seq_cfg

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, seq_cfg


net, feature_utils, seq_cfg = get_model()


@torch.inference_mode()
def video_to_audio(video: gr.Video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    prompt_prefix = sanitize_prompt_prefix(prompt)
    video_save_path = output_dir / f'{prompt_prefix}_{current_time_string}.mp4'
    make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return video_save_path


@torch.inference_mode()
def image_to_audio(image: gr.Image, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    image_info = load_image(image)
    clip_frames = image_info.clip_frames
    sync_frames = image_info.sync_frames
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength,
                      image_input=True)
    audio = audios.float().cpu()[0]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    prompt_prefix = sanitize_prompt_prefix(prompt)
    video_save_path = output_dir / f'{prompt_prefix}_{current_time_string}.mp4'
    video_info = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
    make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return video_save_path


@torch.inference_mode()
def text_to_audio(prompt: str, negative_prompt: str, seed: int, num_steps: int, cfg_strength: float,
                  duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    prompts = [prompt] * TEXT_TO_AUDIO_SAMPLES
    negative_prompts = [negative_prompt] * TEXT_TO_AUDIO_SAMPLES
    audios = generate(clip_frames,
                      sync_frames, prompts,
                      negative_text=negative_prompts,
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    prompt_prefix = sanitize_prompt_prefix(prompt)
    audio_paths = []
    for idx, audio in enumerate(audios.float().cpu(), start=1):
        audio_save_path = output_dir / f'{prompt_prefix}_{current_time_string}_{idx}.flac'
        torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
        audio_paths.append(audio_save_path)
    gc.collect()
    return audio_paths


video_to_audio_tab = gr.Interface(
    fn=video_to_audio,
    description="""
    项目主页：<a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    代码：<a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>

    注意：处理高分辨率视频（短边 >384 px）会更慢，
    但不会提升结果质量。
    """,
    inputs=[
        gr.Video(label='视频'),
        gr.Text(label='提示词'),
        gr.Text(label='负面提示词', value='音乐'),
        gr.Number(label='种子（-1：随机）', value=-1, precision=0, minimum=-1),
        gr.Number(label='步数', value=25, precision=0, minimum=1),
        gr.Number(label='引导强度', value=4.5, minimum=1),
        gr.Number(label='时长（秒）', value=8, minimum=1),
    ],
    outputs=gr.Video(label='输出视频'),
    cache_examples=False,
    title='MMAudio — 视频转音频合成',
    **interface_kwargs,
    examples=[
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_beach.mp4',
            '海浪, 海鸥',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_serpent.mp4',
            '',
            '音乐',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_seahorse.mp4',
            '气泡',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_india.mp4',
            '印度宗教音乐',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_galloping.mp4',
            '马蹄疾驰',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_kraken.mp4',
            '海浪, 暴风雨',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/mochi_storm.mp4',
            '风暴',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_spring.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_typing.mp4',
            '打字声',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_wake_up.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
        [
            'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_nyc.mp4',
            '',
            '',
            0,
            25,
            4.5,
            10,
        ],
    ])

text_to_audio_tab = gr.Interface(
    fn=text_to_audio,
    description="""
    项目主页：<a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    代码：<a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>
    """,
    inputs=[
        gr.Text(label='提示词'),
        gr.Text(label='负面提示词'),
        gr.Number(label='种子（-1：随机）', value=-1, precision=0, minimum=-1),
        gr.Number(label='步数', value=25, precision=0, minimum=1),
        gr.Number(label='引导强度', value=4.5, minimum=1),
        gr.Number(label='时长（秒）', value=8, minimum=1),
    ],
    outputs=[
        make_audio_output(label=f'输出音频 {idx}')
        for idx in range(1, TEXT_TO_AUDIO_SAMPLES + 1)
    ],
    cache_examples=False,
    title='MMAudio — 文本转音频合成',
    **interface_kwargs,
)

image_to_audio_tab = gr.Interface(
    fn=image_to_audio,
    description="""
    项目主页：<a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    代码：<a href="https://github.com/hkchengrex/MMAudio">https://github.com/hkchengrex/MMAudio</a><br>

    注意：处理高分辨率图片（短边 >384 px）会更慢，
    但不会提升结果质量。
    """,
    inputs=[
        gr.Image(type='filepath', label='图片'),
        gr.Text(label='提示词'),
        gr.Text(label='负面提示词'),
        gr.Number(label='种子（-1：随机）', value=-1, precision=0, minimum=-1),
        gr.Number(label='步数', value=25, precision=0, minimum=1),
        gr.Number(label='引导强度', value=4.5, minimum=1),
        gr.Number(label='时长（秒）', value=8, minimum=1),
    ],
    outputs=gr.Video(label='输出视频'),
    cache_examples=False,
    title='MMAudio — 图像转音频合成（实验）',
    **interface_kwargs,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    gr.TabbedInterface([video_to_audio_tab, text_to_audio_tab, image_to_audio_tab],
                       ['视频转音频', '文本转音频', '图像转音频（实验）']).launch(
                           server_port=args.port, allowed_paths=[output_dir])
