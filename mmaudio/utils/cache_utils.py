import os
from pathlib import Path


def configure_local_cache(cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        env_dir = os.environ.get('MMAUDIO_CACHE_DIR')
        cache_dir = Path(env_dir) if env_dir else Path(__file__).resolve().parents[2] / '.cache' / 'mmaudio'

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault('MMAUDIO_CACHE_DIR', str(cache_dir))
    hf_home = cache_dir / 'huggingface'
    os.environ.setdefault('HF_HOME', str(hf_home))
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(hf_home / 'hub'))
    os.environ.setdefault('TORCH_HOME', str(cache_dir / 'torch'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache_dir))
    os.environ.setdefault('OPENCLIP_CACHE_DIR', str(cache_dir / 'open_clip'))

    return cache_dir
