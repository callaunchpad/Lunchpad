import sys
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import pickle


def text_to_image():

    #FIXME: Use StableDiffusion to generate new image
    with open('/home/sebbyzhao/Lunchpad/data/prompt.pt', 'rb') as pickle_file:
        prompt = pickle.load(pickle_file)
    # prompt = pickle.load('home/sebbyzhao/Lunchpad/data/prompt.pt')
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')

    new_image = pipe(prompt).images[0]

    # Save file smwhere.
    with open('/home/sebbyzhao/Lunchpad/data/new_image.pt', 'wb') as pickle_file:
        prompt = pickle.dump(new_image, pickle_file)
    # torch.save(new_image, '/home/sebbyzhao/Lunchpad/data/new/new_image.pt')

if __name__ == '__main__':
    text_to_image()