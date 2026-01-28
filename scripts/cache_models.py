import argparse
import gc
from pathlib import Path

from mmaudio.utils.cache_utils import configure_local_cache

CLIP_MODEL_ID = 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'
BIGVGAN_MODEL_ID = 'nvidia/bigvgan_v2_44khz_128band_512x'


def prefetch_mmaudio_weights() -> None:
    from mmaudio.eval_utils import all_model_cfg

    for cfg in all_model_cfg.values():
        cfg.download_if_needed()


def prefetch_open_clip() -> None:
    import open_clip

    result = open_clip.create_model_from_pretrained(CLIP_MODEL_ID, return_transform=False)
    model = result[0] if isinstance(result, (tuple, list)) else result
    del model
    gc.collect()


def prefetch_bigvgan() -> None:
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2

    model = BigVGANv2.from_pretrained(BIGVGAN_MODEL_ID, use_cuda_kernel=False)
    del model
    gc.collect()


def prefetch_torch_hub() -> None:
    import torch
    from mmaudio.ext.synchformer import vit_helper

    torch.hub.load_state_dict_from_url(vit_helper.default_cfgs['vit_1k'], progress=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',
                        type=Path,
                        default=None,
                        help='Local cache root for HF/Torch/OpenCLIP (default: ./.cache/mmaudio).')
    parser.add_argument('--skip-open-clip', action='store_true', help='Skip OpenCLIP weights.')
    parser.add_argument('--skip-bigvgan', action='store_true', help='Skip BigVGAN v2 weights.')
    parser.add_argument('--skip-torch-hub', action='store_true', help='Skip Torch Hub weights.')
    args = parser.parse_args()

    cache_root = configure_local_cache(args.cache_dir)
    print(f'Using cache dir: {cache_root}')

    prefetch_mmaudio_weights()
    if not args.skip_open_clip:
        prefetch_open_clip()
    if not args.skip_bigvgan:
        prefetch_bigvgan()
    if not args.skip_torch_hub:
        prefetch_torch_hub()

    print('All model files are cached locally.')


if __name__ == '__main__':
    main()
