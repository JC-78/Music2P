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

from transformers import pipeline

from torchvision import transformers
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder

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

def load_image(img_path, image_size, device):
    raw_image = Image.open(img_path).conver('RGB')

    w, h = raw_image.image_size
    # raw_image.show()

    transform = transforms.Compos([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image = transform(raw_image).unsqueeze(0).to(device)

    return image


def BartSummary():

    if os.path.isfile("transfer.pth") == False:
        torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth', 'transfer.pth')
        torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/electronic.mp3', 'electronic.mp3')
        torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/orchestra.wav', 'orchestra.wav')

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

def BlipCap():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    base_path = "/Music2P/custom_data"

    image_path = os.path.join(base_path, img_name)
    image = load_image(img_path=image_path, image_size=img_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=img_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        for i in range(len(caption)):
            print(f"caption [{i}]: {caption[i]}")  



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--img_name', type=str, default='hedgehog.jpg', help='Path to the image')
    args = parser.parse_args()

    BartSummary()
    BlipCap(args.img_name)
