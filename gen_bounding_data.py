from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
from os import path
from os import makedirs
import numpy as np
import glob
import xml.etree.ElementTree as ET

'''
SETUP

Set data_dir to the directory of the game dataset (character/stage files) relative to this script.
If it's in the same directory, you don't need to change this.

Set char_dirs to a list of folder names for the character(s) you want to generate bounding data for.
'''
data_dir = 'game_data'
char_dir = ['ryu']
# char_dir = ['akuma', 'alex', 'chun-li', 'dudley', 'elena', 'hugo', 'ibuki',
#             'ken', 'makoto', 'necro', 'oro', 'q', 'remy', 'ryu', 'sean', 'twelve',
#             'urien', 'yang', 'yun']

def GenerateXML(fileName, Path, Width, Height, Character, Save_path , x_min, y_min, x_max, y_max) : 
      
    root = ET.Element("annotation") 

    folder = ET.SubElement(root, "folder")
    folder.text = "all" 

    filename = ET.SubElement(root, "filename")
    filename.text = fileName + ".png"

    path = ET.SubElement(root, "path") #full path
    path.text = Path + fileName + ".png"

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(Width) 
    height = ET.SubElement(size, "height") 
    height.text = str(Height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    obj = ET.SubElement(root, "object")
    name = ET.SubElement(obj, "name")
    name.text = Character
    pose = ET.SubElement(obj, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(obj, "truncated")
    truncated.text = "1"
    difficult = ET.SubElement(obj, "difficult")
    difficult.text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin") #left
    xmin.text = str(x_min)
    ymin = ET.SubElement(bndbox, "ymin") #bottom
    ymin.text = str(y_max)
    xmax = ET.SubElement(bndbox, "xmax") #right
    xmax.text = str(x_max)
    ymax = ET.SubElement(bndbox, "ymax") #top
    ymax.text = str(y_min)

    tree = ET.ElementTree(root)
    tree.write(Save_path + fileName + ".xml")

def select_random_dir(root):
    candidates = glob.glob("{}/*".format(root))
    ind = np.random.randint(0, len(candidates))
    return candidates[ind]

def gif_to_frames(img):
    frames = []
    frame = Image.new('RGBA', img.size)
    frame.paste(img, (0,0), img.convert('RGBA'))
    bbox = frame.getbbox()
    frame = frame.crop(bbox)
    frames.append(frame)
    for _ in range((img.n_frames)//2-1):
        img.seek(img.tell() + 2)
        frame = Image.new('RGBA', img.size)
        frame.paste(img, (0,0), img.convert('RGBA'))
        bbox = frame.getbbox()
        frame = frame.crop(bbox)
        if np.random.rand() > 0.5:
            frame = frame.transpose(Image.FLIP_LEFT_RIGHT) # randomly flip image to account for both sides
        frames.append(frame)
    return frames

def generate_bounds(character_path, example_idx):
    # Select a random move and color
    category_path = select_random_dir(character_path)
    while category_path == path.join(character_path, 'Entrance'): # exclude entrance animations, Ryu's includes Ken
        category_path = select_random_dir(character_path)
    # Select a random move
    move_path = select_random_dir(category_path)
    move = Image.open(move_path)

    # Extract individual frames from move gif
    frames = gif_to_frames(move)

    for frame in frames:
        # Select a random stage
        stage_path = select_random_dir(path.join(path.dirname(__file__), data_dir, 'stages'))
        stage = Image.open(stage_path)
        stage_width, stage_height = stage.size

        # Get dimensions of character for overlaying onto the stage later
        char_width, char_height = frame.size

        height_offset = stage_height-char_height - 20
        width_offset = int((stage_width-char_width) * np.random.rand(1))
        if (width_offset + char_width) > stage_width: # prevent out of bounds
            width_offset = stage_width - char_width
        
        print(char_width, char_height)
        GenerateXML(str(example_idx), '', char_width, char_height, character, data_save_path, width_offset, height_offset, width_offset+char_width, height_offset-char_height)

        # combine stage with character and crop to character's bounds
        stage_copy = stage.copy()
        stage_copy.paste(frame, (width_offset, height_offset), frame.convert('RGBA'))
        stage_copy.save("{}/{}.png".format(data_save_path, example_idx))

        example_idx += 1
        if example_idx == num_examples:
            break

    return example_idx

bounding_data_path = path.join(path.dirname(__file__), 'bounding_data')
num_examples = 20 # number of training examples to generate
example_idx = 0
for character in char_dir:
    while example_idx < num_examples:
        character_path = path.join(path.dirname(__file__), data_dir, 'characters', character)
        data_save_path = path.join(bounding_data_path, character)
        makedirs(data_save_path, exist_ok=True)
        example_idx = generate_bounds(character_path, example_idx)