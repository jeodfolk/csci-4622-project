from PIL import Image, ImageSequence
import glob
import os
from matplotlib import pyplot as plt
import numpy as np

def select_random_dir(root):
    candidates = glob.glob("{}/*".format(root))
    ind = np.random.randint(0, len(candidates))
    return candidates[ind]

def generate_seq():
    # Select a random stage
    stagepath = select_random_dir('stages')
    stage = Image.open(stagepath)
    stage_width, stage_height = stage.size

    # Select a random color
    color = select_random_dir('ryu-png')

    # Generate sequence of moves
    moves = []
    num_moves = 3
    for _ in range(num_moves):
        # Select a random category
        category = select_random_dir(color)
        while category == os.path.join(color, 'Entrance'): # don't use entrance for now, it includes Ken in one
            category = select_random_dir(color)
        # Select a random move
        move = select_random_dir(category)
        num_frames = len(glob.glob("{}/*".format(move)))

        # Randomly limit the number of frames between half to all to simulate early cancellation
        limiter_threshold = 0.8
        frame_lim = None
        if np.random.rand(1) > limiter_threshold:
            frame_lim = np.random.randint(num_frames//2, num_frames)
        else:
            frame_lim = num_frames

        moves.append([move, frame_lim, os.path.basename(category)])

    # Use the sequence of the label as the name for the generated directory
    # Category1^Move1^NFrames_Category2^Move2^NFrames_...CategoryN^MoveN^NFrames
    # where NFrames is the number of frames attributed to that move
    movedirs = [[x[2], os.path.basename(x[0]), x[1]] for x in moves]
    label = ''
    for category, move, lim in movedirs:
        label += '{}^{}^{}_'.format(category, move, lim)
    label = label[:-1] # remove trailing underscore
    sequence_path = os.path.relpath("sequences/{}".format(label))
    # It's possible to generate the same sequence twice
    # However, they're still unique (by colors / stage location / other alterations), so we still want to store it
    # Generate a random unique number at the end just to keep them as separate data points
    if os.path.isdir(sequence_path):
        new_path = sequence_path + '#'
        while os.path.isdir(sequence_path):
            sequence_path = new_path + str(np.randint(1000000, 9999999))
    os.makedirs(sequence_path)

    # initialize width offset here for special handling across moves
    # try to keep moves on the same offset, but if wider moves go off-screen, adjust them more left
    initial_width_offset = None 
    sequence_frame_num = 0
    for move, frame_lim, _ in moves:
        frames = glob.glob("{}/*.png".format(move))
        # Sort by frame number (filename needs to be int, not string)
        frames = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Save all frames of gif as individual png in new folder
        move_frame_num = 0
        for frame in frames:
            character = Image.open(frame)
            # Get dimensions of character for overlaying onto the stage later
            char_width, char_height = character.size

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

            stage_copy = stage.copy()

            # combine stage with character and crop to character's bounds
            stage_copy.paste(character, (width_offset, height_offset), character.convert('RGBA'))
            stage_copy = stage_copy.crop((width_offset, height_offset, width_offset+char_width, height_offset+char_height))
            stage_copy = stage_copy.resize((256, 256))

            stage_copy.save("{}/{}.png".format(sequence_path, sequence_frame_num))

            sequence_frame_num += 1

            move_frame_num += 1
            if move_frame_num == frame_lim: # break when frame limit for this move is reached
                break


# set to any number to generate more data
for _ in range(100):
    generate_seq()