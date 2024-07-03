import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from diffusers.utils import load_image
import glob
import cv2 as cv
import torchvision
import logging
import einops
import os.path as osp
from glob import glob
# from filesystem import scan_dir

logger = logging.getLogger(__name__)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_ddim_latents_at_t(t, ddim_latents_path):
    ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path, map_location="cpu")
    logger.debug(f"Loaded ddim_latents_at_t from {ddim_latents_at_t_path}")
    return ddim_latents_at_t


def load_ddim_latents_at_T(ddim_latents_path):
    noisest = max(
        [int(x.split("_")[-1].split(".")[0]) for x in glob.glob(os.path.join(ddim_latents_path, f"ddim_latents_*.pt"))]
    )
    ddim_latents_at_T_path = os.path.join(ddim_latents_path, f"ddim_latents_{noisest}.pt")
    ddim_latents_at_T = torch.load(ddim_latents_at_T_path)  # [b, c, f, h, w] [1, 4, 16, 40, 64]
    return ddim_latents_at_T


# Modified from tokenflow/utils.py
def convert_video_to_frames(video_path, img_size=(512, 512), save_frames=True):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith(".mov"):
        video = T.functional.rotate(video, -90)
    if save_frames:
        video_name = Path(video_path).stem
        video_dir = Path(video_path).parent
        os.makedirs(f"{video_dir}/{video_name}", exist_ok=True)
    frames = []
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        logger.info(f"Original video frame size: {image.size}")
        if image.size != img_size:
            image_resized = image.resize(img_size, resample=Image.Resampling.LANCZOS)
            logger.info(f"Resized video frame, height, width: {image_resized.size}, {img_size[1]}, {img_size[0]}")
        else:
            image_resized = image
        if save_frames:
            image_resized.save(f"{video_dir}/{video_name}/{ind}.png")
            print(f"Saved frame {video_dir}/{video_name}/{ind}.png")
        frames.append(image_resized)
    return frames


# Modified from tokenflow/utils.py
def load_video_frames(frames_path, n_frames, image_size=(512, 512)):
    # Load paths
    count, paths = scan_dir(frames_path)
    paths.sort(key=lambda p: int(os.path.basename(p).split(".")[0]))
    paths = paths[:n_frames]
    # paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
    frames = [load_image(p) for p in paths]
    # Check if the frames are the right size
    for i, f in enumerate(frames):
        if f.size != image_size:
            # logger.error(f"Frame size {f.size} does not match config.image_size {image_size}")
            # raise ValueError(f"Frame size {f.size} does not match config.image_size {image_size}")
            frames[i] = f.resize(image_size, resample=Image.Resampling.LANCZOS)
    return paths, frames


def mask_preprocess_static(mask: str, device, dtype, batch_size, channel, frames, downscale):
    mask = Image.open(mask).convert("L")
    mask_hw = mask.size
    mask = mask.resize((mask_hw[0] // downscale, mask_hw[1] // downscale))
    # if binary:
    mask_cv = np.asarray(mask)
    _, mask_cv = cv.threshold(mask_cv, 10, 255, cv.THRESH_BINARY)
    mask_binary = Image.fromarray(mask_cv)

    mask_transform = torchvision.transforms.PILToTensor()

    mask_tensor = mask_transform(mask)
    mask_binary_tensor = mask_transform(mask_binary)

    mask_tensor = mask_tensor.to(torch.float32).div_(255.0).to(device, dtype).squeeze(0)
    mask_binary_tensor = mask_binary_tensor.to(torch.float32).div_(255.0).to(device, torch.bool).squeeze(0)

    return einops.repeat(mask_tensor, "h w -> b c t h w", b=batch_size, c=channel, t=frames), \
           einops.repeat(mask_binary_tensor, "h w -> b c t h w", b=batch_size, c=channel, t=frames)


def mask_preprocess_dynamic(mask: str, device, dtype, batch_size, channel, frames, downscale):
    mask_paths = glob(osp.join(mask, "*.png"))
    # assert len(mask_paths) == frames

    mask_paths.sort(key=lambda p: int(osp.basename(p).split(".")[0]))
    if len(mask_paths) != frames:
        mask_paths = mask_paths[:frames]

    mask_tensor_list = []
    mask_binary_tensor_list = []
    for i, maskp in enumerate(mask_paths):
        mask = Image.open(maskp).convert("L")
        mask_hw = mask.size
        mask = mask.resize((mask_hw[0] // downscale, mask_hw[1] // downscale))

        mask_transform = torchvision.transforms.PILToTensor()

        mask_cv = np.asarray(mask)
        _, mask_cv = cv.threshold(mask_cv, 10, 255, cv.THRESH_BINARY)
        mask_binary = Image.fromarray(mask_cv)

        mask_tensor = mask_transform(mask)
        mask_binary_tensor = mask_transform(mask_binary)

        mask_tensor = mask_tensor.to(torch.float32).div_(255.0).to(device, dtype).squeeze(0)
        mask_binary_tensor = mask_binary_tensor.to(torch.float32).div_(255.0).to(device, torch.bool).squeeze(0)
        mask_tensor_list.append(mask_tensor)
        mask_binary_tensor_list.append(mask_binary_tensor)

    mask_tensor = torch.stack(mask_tensor_list, dim=0)
    mask_binary_tensor = torch.stack(mask_binary_tensor_list, dim=0)
    return einops.repeat(mask_tensor, "t h w -> b c t h w", b=batch_size, c=channel), \
           einops.repeat(mask_binary_tensor, "t h w -> b c t h w", b=batch_size, c=channel)


def mask_preprocess(mask: str, device, dtype, batch_size, channel, frames, downscale=8):
    if osp.isdir(mask):
        mas_process_func = mask_preprocess_dynamic
        # return mask_preprocess_dynamic(mask, device, dtype, batch_size, channel, frames, downscale, binary)
    else:
        mas_process_func = mask_preprocess_static
    return mas_process_func(mask, device, dtype, batch_size, channel, frames, downscale)


def cvt_cv_aff2torch_aff(opencv_theta: np.ndarray,
                         src_h: int, src_w: int,
                         dst_h: int, dst_w: int,
                         dtype=torch.float32,
                         device="cpu"):
    m = np.concatenate([opencv_theta, np.array([[0., 0., 1.]], dtype=np.float32)])
    m_inv = np.linalg.inv(m)

    a = np.array([[2 / (src_w - 1), 0., -1.],
                  [0., 2 / (src_h - 1), -1.],
                  [0., 0., 1.]], dtype=np.float32)

    b = np.array([[2 / (dst_w - 1), 0., -1.],
                  [0., 2 / (dst_h - 1), -1.],
                  [0., 0., 1.]], dtype=np.float32)
    b_inv = np.linalg.inv(b)

    pytorch_m = a @ m_inv @ b_inv
    return torch.as_tensor(pytorch_m[:2], dtype=dtype, device=device)


def warp_affine_torch(tensor: torch.Tensor, theta: torch.Tensor, dsize: torch.Size, align_corners=False):
    grid = F.affine_grid(theta, dsize, align_corners=align_corners)
    return F.grid_sample(tensor, grid, align_corners=align_corners, mode="nearest")
