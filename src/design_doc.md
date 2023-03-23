# Design Doc

## 1. Overview
There are a couple of goals:
1. Create pipeline for hardcoded StableDiffusion prompt using name of food + list of ingredients, then pass into inversecooking model
2. Directly use guided image-to-image StableDiffusion to do style transfer, run inference using inversecooking model
3. Add in option to add other ingredients into StableDiffusion prompt
4. Add in option to add both closest and farthest ingredient in latent space.
5. Understand and visualize latent space that is learned from ingredient decoder.

## 2. Files

`utils.py`: Contains all of the utility functions. Preprocessing, file saving/model loading, and anything regarding data format.

`main.py`: House CLI way of conducting model inference.

`prompt.py`: Defines class that generates the prompt to pass into StableDiffusion. Should be generalizable for range of prompts.

`eda.ipynb`: All of the exploratory data analysis.


## 2. Functions 

`utils.py`
- A 

`main.py`
- A

`prompt.py`
- A

`eda.ipynb`
- A

