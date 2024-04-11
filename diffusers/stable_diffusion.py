import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import pickle
from PIL import Image
import os

root_dir = os.path.abspath(os.getcwd())
data_dir = 'data'

def text_to_image():

    #FIXME: Use StableDiffusion to generate new image
    with open(os.path.join(root_dir, data_dir, 'prompt.pt'), 'rb') as pickle_file:
        prompt = pickle.load(pickle_file)
    # prompt = pickle.load('home/sebbyzhao/Lunchpad/data/prompt.pt')
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')

    new_image = pipe(prompt).images[0]

    # Save file smwhere.
    new_image.save('data/new_image.jpg')

def img_to_img():

    #FIXME: Use StableDiffusion to generate new image
    with open(os.path.join(root_dir, data_dir, 'prompt.pt'), 'rb') as pickle_file:
        prompt = pickle.load(pickle_file)
        image = pickle.load(pickle_file)
    # prompt = pickle.load('home/sebbyzhao/Lunchpad/data/prompt.pt')
    
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16).to('cuda')
    images = pipe(prompt=prompt, image=image, strength=0.8, guidance_scale=7.5).images
    new_image = images[0]

    # Save file smwhere.
    new_image.save('data/new_image.jpg')

if __name__ == '__main__':
    text_to_image()