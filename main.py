import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import requests
from io import BytesIO

#FIXME: Set up argparser + arguments

parser.add_command(parser, "-i", "--image", metavar="IMAGE_FLIP", help = "Path to your input image")

def main():

    data_dir = '../data' #path to vocab and model checkpoint

    use_gpu = True #running on gpu or cpu
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'


    ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
    vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

    ingr_vocab_size = len(ingrs_vocab)
    instrs_vocab_size = len(vocab)

    t = time.time()
    import sys; sys.argv=['']; del sys
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only=False
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    # Load the trained model parameters
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    model.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.to(device)
    model.eval()
    model.ingrs_only = False
    model.recipe_only = False

    image_path = args['image'] #inputted image path
    image = Image.open(image_path).convert('RGB') #image

    transf_list_batch = []             #variables 
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    to_input_transf = transforms.Compose(transf_list_batch)

    greedy = [True, False, False, False]
    beam = [-1, -1, -1, -1]
    temperature = 1.0
    numgens = len(greedy)

    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)

    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    #FIXME: Run model inference on original image

    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=greedy[i], 
                temperature=temperature, beam=beam[i], true_ingrs=None)
                
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
                
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
            
    recipe_name = outs['title']
    ingredients = outs['ingrs']

    option = input("Do you want to use the text to image option (1) or the image to image option (2)?")
    if (option == "1"):
        #FIXME: Create hardcoded StableDiffusion prompt   

        prompt = "Fancy food plating of " + recipe_name + " with ingredients " + ingredients

        #FIXME: Use StableDiffusion to generate new image

        pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')

        new_image = pipe(prompt).images[0]

        #FIXME: Run model inference on new image

        image_transf_new = transform(new_image)
        image_tensor_new = to_input_transf(image_transf_new).unsqueeze(0).to(device)

        for i in range(numgens):
            with torch.no_grad():
                outputs = model.sample(image_tensor_new, greedy=greedy[i], 
                                    temperature=temperature, beam=beam[i], true_ingrs=None)
                
            ingr_ids = outputs['ingr_ids'].cpu().numpy()
            recipe_ids = outputs['recipe_ids'].cpu().numpy()
                
            outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
            
            recipe_name_new = outs['title']
            ingredients_new = outs['ingrs']
            recipe = outs['recipe']


        #FIXME: Print out output/display new image onto terminal

        plt.imshow(image_transf_new)
        plt.axis('off')
        plt.show()
        plt.close()

        print('TITLE:',recipe_name_new)
        print('\nINGREDIENTS:')
        print(', '.join(ingredients_new))
        print('\nINSTRUCTIONS:')
        print('-' + '\n-'.join(recipe))

    elif (option == "2"):

        # all the img2img stuff 

        model_id_or_path = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        prompt = "A fancy plating of this " + recipe_name
        images = pipe(prompt=prompt, image=image, strength=0.8, guidance_scale=7.5).images
        fancy_img = images[0]

        # now take that fancy image and run it through inv cooking again

        fancy_img_transf = transform(fancy_img)
        fancy_img_tensor = to_input_transf(fancy_img_transf).unsqueeze(0).to(device)

        for i in range(numgens):
            with torch.no_grad():
                outputs = model.sample(fancy_img_tensor, greedy=greedy[i], 
                                    temperature=temperature, beam=beam[i], true_ingrs=None)
                
            ingr_ids = outputs['ingr_ids'].cpu().numpy()
            recipe_ids = outputs['recipe_ids'].cpu().numpy()
                
            outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
            
            new_recipe_name = outs['title']
            new_ingredients = outs['ingrs']
            new_recipe = outs['recipe']

        # print that shit

        plt.imshow(fancy_img_transf)
        plt.axis('off')
        plt.show()
        plt.close()

        print('TITLE:',new_recipe_name)
        print('\nINGREDIENTS:')
        print(', '.join(new_ingredients))
        print('\nINSTRUCTIONS:')
        print('-' + '\n-'.join(new_recipe))

    