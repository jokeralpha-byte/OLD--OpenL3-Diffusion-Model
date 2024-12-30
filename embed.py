import os
import warnings

import numpy as np
import torch
import core
from openl3model import OpenL3Embedding

import audio_utils as au

def _check_device(device: int):
    assert isinstance(device, int)
    assert device >= 0#检查一下DEVICE

def embed(model: OpenL3Embedding, audio: np.ndarray, sample_rate: int, hop_size: int = 1, device: int = None):#采样函数
    """compute OpenL3 embeddings for a given audio array. 
    Args:
        model (OpenL3Embedding): OpenL3 model to use
        audio (np.ndarray): audio array with shape (channels, samples). 
                            audio will be downmixed to mono. 
        sample_rate (int): input sample rate. Will be resampled to torchopenl3.SAMPLE_RATE
        hop_size (int, optional): hop size, in seconds. Defaults to 1.
    Returns:
        np.ndarray: embeddings with shape (frame, features)
    """
    au.core._check_audio_types(audio)
    if au.core._is_zero(audio):
        warnings.warn(f'provided audio array is all zeros')
    # resample, downmix, and zero pad if needed
    audio = au.resample(audio, sample_rate, core.SAMPLE_RATE)
    audio = au.downmix(audio)

    # split audio into overlapping windows as dictated by hop_size
    hop_len: int = hop_size * core.SAMPLE_RATE
    audio = au.window(audio, window_len=1*core.SAMPLE_RATE, hop_len=hop_len)

    # convert to torch tensor!
    audio = torch.from_numpy(audio)

    # GPU support
    if device is not None:
        model = model.to(device)
        audio = audio.to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model(audio)
    
    return embeddings.cpu().numpy()

def embed_from_file_to_array(model: OpenL3Embedding, path_to_audio: str, hop_size: int = 1):
    """compute OpenL3 embeddings from a given audio file

    Args:
        model (OpenL3Embedding): model to embed with
        path_to_audio (str): path to audio file
        hop_size (int, optional): embedding hop size, in seconds. Defaults to 1.

    Returns:
        np.ndarray: embeddings with shape (frame, features)
    """
    # load audio
    audio = au.load_audio_file(path_to_audio, sample_rate=core.SAMPLE_RATE)
    return embed(model, audio, core.SAMPLE_RATE, hop_size)

def embed_from_file_to_file(model: OpenL3Embedding, path_to_audio: str, path_to_output: str, hop_size: int = 1):
    """compute OpenL3 embeddings from a given audio file and save to an output path in .npy format

    Args:
        model (OpenL3Embedding): model to embed with
        path_to_audio (str): path to audio file
        path_to_output (str): path to output (.npy) file
        hop_size (int, optional): embedding hop size, in seconds. Defaults to 1.
    """
    # get embedding array
    embeddings = embed_from_file_to_array(model, path_to_audio, hop_size)
    np.save(path_to_output, embeddings)