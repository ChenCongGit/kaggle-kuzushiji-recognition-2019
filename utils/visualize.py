from PIL import Image, ImageDraw, ImageFont
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

fontsize = 50

font = ImageFont.truetype('./data/record/NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('./data/kuzushiji-recognition/unicode_translation.csv').values}

# This function takes in a filename of an image, and the labels in the string format given in a submission csv, and returns an image with the characters and predictions annotated.
def visualize_predictions(image_fn, labels, save_path):
    # Convert annotation string to array
    labels = np.array(labels.split(' ')).reshape(-1, 3)
    
    # Read image
    imsource = Image.open(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y in labels:
        x, y = int(x), int(y)
        char = unicode_map[codepoint] # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x-10, y-10, x+10, y+10), fill=(255, 0, 0, 255))
        char_draw.text((x+25, y-fontsize*(3/4)), char, fill=(255, 0, 0, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    print(image_fn, save_path)
    imsource.save(save_path, quality = 30)

df_test = pd.read_csv('./result/submission/test/submission_test.csv')
VIS_SAVE_DIR = './result/submission_vis/'

for i in range(len(df_test)):
    if i <= 100:
        img, labels = df_test.values[i]
        if str(labels) != 'nan':
            viz = visualize_predictions('./data/kuzushiji-recognition/test_images/{}.jpg'.format(img), labels[:-1], VIS_SAVE_DIR+img+'.jpg')
    

