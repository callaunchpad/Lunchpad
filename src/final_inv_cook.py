import pkg_resources
pkg_resources.require("torch==0.4.1")
import torch
import numpy as np
import os
import argparse
from argparse import Namespace
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time

def final_inv_cook():
    root_dir = os.path.abspath(os.getcwd())
    data_dir = 'data' #path to vocab and model checkpoint

    with open(os.path.join(root_dir, data_dir, 'ingredients.npy'), 'rb') as f:
        ingredients = np.load(f)
    ingredients = torch.from_numpy(ingredients)
    new_image = Image.open(os.path.join(root_dir, data_dir, 'new_image.jpg'))

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
    image_transf_new = transform(new_image)
    image_tensor_new = to_input_transf(image_transf_new).unsqueeze(0).to(device)

    ingredients = ingredients.to(device)

    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(image_tensor_new, greedy=greedy[i], 
                                   temperature=temperature, beam=beam[i], true_ingrs=ingredients)
            
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
            
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
        
        recipe_name_new = outs['title']
        ingredients_new = outs['ingrs']
        recipe = outs['recipe']


    print('TITLE:',recipe_name_new)
    print('\nINGREDIENTS:')
    print(', '.join(ingredients_new))
    print('\nINSTRUCTIONS:')
    print('-' + '\n-'.join(recipe))

if __name__ == '__main__':
    final_inv_cook()