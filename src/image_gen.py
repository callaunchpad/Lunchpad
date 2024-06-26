import pkg_resources
pkg_resources.require("torch==0.4.1")
import torch
import numpy as np
import os
import pickle
import argparse
from argparse import Namespace
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time

#FIXME: Set up argparser + arguments

parser = argparse.ArgumentParser(
                    prog='Lunchpad',
                    description='Command line usage for Lunchpad')
                
parser.add_argument("-i", "--image", metavar="IMAGE_FLIP", help = "Path to your input image")
parser.add_argument("--save_images", help="Directory to save generated images")

def image_gen(arg):
    root_dir = os.path.abspath(os.getcwd())
    data_dir = 'data' #path to vocab and model checkpoint

    use_gpu = True #running on gpu or cpu
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'


    ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
    vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

    ingr_vocab_size = len(ingrs_vocab)
    instrs_vocab_size = len(vocab)

    t = time.time()
    import sys; sys.argv=['']; del sys

    # Hardcoded args
    args = Namespace(aux_data_dir='../data', batch_size=128, beam=-1, crop_size=224, decay_lr=True, dropout_decoder_i=0.3, dropout_decoder_r=0.3, dropout_encoder=0.3, embed_size=512, es_metric='loss', eval_split='val', finetune_after=-1, get_perplexity=False, greedy=False, image_model='resnet50', image_size=256, ingrs_only=True, label_smoothing_ingr=0.1, learning_rate=0.001, log_step=10, log_term=False, loss_weight=[1.0, 0.0, 0.0, 0.0], lr_decay_every=1, lr_decay_rate=0.99, max_eval=4096, maxnumims=5, maxnuminstrs=10, maxnumlabels=20, maxseqlen=15, model_name='model', n_att=8, n_att_ingrs=4, num_epochs=400, num_workers=8, numgens=3, patience=50, project_name='inversecooking', recipe1m_dir='path/to/recipe1m', recipe_only=False, resume=False, save_dir='path/to/save/models', scale_learning_rate_cnn=0.01, suff='', temperature=1.0, tensorboard=True, transf_layers=16, transf_layers_ingrs=4, transfer_from='', use_lmdb=True, use_true_ingrs=False, weight_decay=0.0)
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

    image_path = arg.image #inputted image path
    image = Image.open(image_path).convert('RGB') #image

    transf_list_batch = []             #variables 
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    to_input_transf = transforms.Compose(transf_list_batch)

    greedy = [True, False, False, False]
    beam = [-1, -1, -1, -1]
    temperature = 1.0
    numgens = 1 #len(greedy)

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
        ingredients = outs['ingrs'] # ingredient list


    #FIXME: Create hardcoded StableDiffusion prompt

    ingredients = ', '.join(ingredients)
    prompt = "Fancy food plating of " + recipe_name + " with ingredients " + ingredients
    print(prompt)

    # goal: save ingredient latent vector
    with open(os.path.join(root_dir, data_dir, 'ingredients.npy'), 'wb') as f:
        np.save(f, ingr_ids)
    
    with open(os.path.join(root_dir, data_dir, 'prompt.pt'), 'wb') as pickle_file:
        pickle.dump(prompt, pickle_file)
        pickle.dump(ingredients, pickle_file)
        pickle.dump(image, pickle_file)
    

    

if __name__ == '__main__':
    arg = parser.parse_args()
    image_gen(arg)