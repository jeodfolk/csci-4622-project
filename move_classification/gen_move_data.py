from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
from os import path
from os import makedirs
from math import ceil
import numpy as np
import glob

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
    for _ in range(img.n_frames-1):
        img.seek(img.tell() + 1)
        frame = Image.new('RGBA', img.size)
        frame.paste(img, (0,0), img.convert('RGBA'))
        bbox = frame.getbbox()
        frame = frame.crop(bbox)
        frames.append(frame)
    return frames

def select_random_frames(frames, label, num_frames):
    if label:
        num_frames = ceil(np.random.rand()*num_frames+1.00001)
        frames = frames[:num_frames]
    else:
        num_frames = ceil((np.random.rand()+0.5)*len(frames))
        frames = frames[:num_frames]
    return frames

def generate_sequence(character_path, example_idx, num_frames):
    # Select a random stage
    stage_path = select_random_dir(path.join(path.dirname(__file__), data_dir, 'stages'))
    stage = Image.open(stage_path)
    stage_width, stage_height = stage.size

    # Select a random move category
    category_path = select_random_dir(character_path)
    while category_path == path.join(character_path, 'Entrance'): # exclude entrance animations, Ryu's includes Ken
        category_path = select_random_dir(character_path)
    # Select a random move
    move_path = select_random_dir(category_path)
    move_img = Image.open(move_path)

    character_name = path.basename(character_path)
    category_name = path.basename(category_path)
    move_name = path.basename(move_path) # remove file extension
    target_color = move_name[:move_name.find('-')] # select from one color to avoid duplicate moves
    move_name = move_name[move_name.find('-')+1:-4] # exclude color and file extension
    label = '{}-{}-{}'.format(character_name, category_name, move_name)
    # Ensure all other moves match in color

    # Extract individual frames from move gif
    frames = select_random_frames(gif_to_frames(move_img), True, num_frames)

    sequence = frames # Overall frame sequence for the move sample
    if len(sequence) > num_frames: # if initial move too long, crop it
        sequence = sequence[:num_frames]
    while len(sequence) < num_frames:
        # Select a random move category
        category_path = select_random_dir(character_path)
        while category_path == path.join(character_path, 'Entrance'): # exclude entrance animations, Ryu's includes Ken
            category_path = select_random_dir(character_path)
        # Select a random move
        move_path = select_random_dir(category_path)
        # Convert move to target color
        move_name = path.basename(move_path)
        move_name = target_color + move_name[move_name.find('-'):]
        move_path = path.join(path.dirname(move_path), move_name)
        move_img = Image.open(move_path)
        # Extract individual frames from move gif
        frames = select_random_frames(gif_to_frames(move_img), False, num_frames)
        # Add new frames to sequence
        sequence = frames + sequence
        # If too many frames were just added, crop them
        if len(sequence) > num_frames:
            sequence = sequence[len(sequence)-num_frames:]

    # initialize width offset here for special handling across moves
    # try to keep moves on the same offset, but if wider moves go off-screen, adjust them more left
    initial_width_offset = None
    sequence_on_stage = []
    for frame in sequence:
        # Get dimensions of character for overlaying onto the stage later
        char_width, char_height = frame.size

        # Set the placement location of the character onto the stage
        # offset from the left
        if initial_width_offset == None: # preserve initial width in case it's altered by an overflowing wide move
            width_offset = int((stage_width-char_width) * np.random.rand(1))
            initial_width_offset = width_offset
        width_offset = initial_width_offset # reset width in case it was altered
        if (width_offset + char_width) > stage_width:
            width_offset = stage_width - char_width
        # offset from the top
        height_offset = stage_height-char_height - 20

        # combine stage with character and crop to character's bounds
        stage_copy = stage.copy()
        stage_copy.paste(frame, (width_offset, height_offset), frame.convert('RGBA'))
        stage_copy = stage_copy.crop((width_offset, height_offset, width_offset+char_width, height_offset+char_height))
        stage_copy = stage_copy.resize((256, 256))
        sequence_on_stage.append(stage_copy)

    for i in range(len(sequence_on_stage)):
        temp = sequence_on_stage[i].convert('RGB')
        save_path = path.join(bounding_data_path, label, str(example_idx))
        makedirs(save_path, exist_ok=True)
        sequence_on_stage[i].save('{}/{}.png'.format(save_path, i))

    example_idx += 1
    return example_idx

bounding_data_path = path.join(path.dirname(__file__), 'move_data')
num_examples = 50000 # number of training examples to generate
num_frames = 6 # number of frames per sequence, with the last frame always matching the label
example_idx = 0
for character in char_dir:
    while example_idx < num_examples:
        character_path = path.join(path.dirname(__file__), data_dir, 'characters', character)
        example_idx = generate_sequence(character_path, example_idx, num_frames)