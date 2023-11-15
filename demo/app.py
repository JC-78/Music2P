import os
import argparse
import gradio as gr
from timeit import default_timer as timer
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from model.bart import BartCaptionModel
from utils.audio_utils import load_audio, STR_CH_FIRST

if os.path.isfile("transfer.pth") == False:
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth', 'transfer.pth')
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/electronic.mp3', 'electronic.mp3')
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/orchestra.wav', 'orchestra.wav')

import os

music_dir = 'custom_data'
music_li = []

for file in os.listdir(music_dir):
    if file.endswith(".mp3"):
        file_path = os.path.join(music_dir, file)
        music_li.append(file_path)


# music = '/mnt/nas/jieun/pop2piano_data/KhBfF_BpfUg.wav'
music = 'custom_data/NewJeans-ditto.mp3'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

example_list = ['electronic.mp3', 'orchestra.wav']
model = BartCaptionModel(max_length = 128)
pretrained_object = torch.load('./transfer.pth', map_location='cpu')
state_dict = pretrained_object['state_dict']
model.load_state_dict(state_dict)
if torch.cuda.is_available():
    torch.cuda.set_device(device)
model = model.cuda(device)
model.eval()

def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

def captioning(audio_path):
    audio_tensor = get_audio(audio_path = audio_path)
    if device is not None:
        audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        output = model.generate(
            samples=audio_tensor,
            num_beams=5,
        )
    inference = ""
    number_of_chunks = range(audio_tensor.shape[0])
    gen_text_li = []
    for chunk, text in zip(number_of_chunks, output):
        time = f"[{chunk * 10}:00-{(chunk + 1) * 10}:00]"
        inference += f"{time}\n{text} \n \n"
        gen_text_li.append(text)
    concat_text = ''.join(gen_text_li)    

    return inference, concat_text

from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", truncation=True)
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':1024,'return_tensors':'pt'}

for music in music_li:
    print(f'----music {music}----')
    infer, concat_text = captioning(music)
    # save code
    print(summarizer(concat_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text'])
    # print(infer)

# infer, concat_text = captioning(music)
# print(infer)
# print(concat_text)

### BART SUMMARIZATION MODEL ###



