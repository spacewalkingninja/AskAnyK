from decord import VideoReader
from decord import cpu
import numpy as np
import glob
import os
import torch
from PIL import Image


import torchvision.transforms as transforms
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)


def loadvideo_decord(sample, sample_rate_scale=1,new_width=384, new_height=384, clip_len=8, frame_sample_rate=2,num_segment=1):
    fname = sample
    vr = VideoReader(fname, width=new_width, height=new_height,
                                     num_threads=1, ctx=cpu(0))
    # handle temporal segments
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) //num_segment
    duration = max(len(vr) // vr.get_avg_fps(),8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

def loadvideo_decord_origin(sample, sample_rate_scale=1,new_width=384, new_height=384, clip_len=8, frame_sample_rate=2,num_segment=1):
    fname = sample
    vr = VideoReader(fname, 
                                     num_threads=1, ctx=cpu(0))
    # handle temporal segments
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) //num_segment
    duration = max(len(vr) // vr.get_avg_fps(),8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


def load_images_decord(folder_path, image_size=(384, 384), normalize=True):
    """Loads and preprocesses images from a folder using PyTorch and torchvision transforms.

    Args:
        folder_path (str): Path to the folder containing images.
        image_size (tuple): Desired size of the image in pixels (width, height).
        normalize (bool): If True, normalize the pixel values of the image.

    Returns:
        list of numpy.ndarray: A list of preprocessed image tensors.
    """
    # Define preprocessing transforms
    preprocess = transforms.Compose([
    #    transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    if normalize:
        preprocess = transforms.Compose([
#            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    images = []
    for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
        image = Image.open(filename).convert('RGB').resize(image_size)
        image_tensor = preprocess(image)
        if image_tensor.shape[0] == 3:
            alpha_tensor = torch.ones_like(image_tensor[0]).unsqueeze(0)
            image_tensor = torch.cat([image_tensor, alpha_tensor], dim=0)
        images.append(image_tensor.numpy())

    return images