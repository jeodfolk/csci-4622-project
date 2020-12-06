import pickle
import glob
from os import path
from os import makedirs

move_to_idx = {}
idx_to_move = {}

characters = glob.glob('{}/*'.format(path.join(path.dirname(__file__), 'game_data', 'characters')))
characters = ['move_classification/game_data/characters/ryu']

cur_idx = 0
for character in characters:
    character_name = path.basename(character)
    categories = glob.glob('{}/*'.format(character))
    for category in categories:
        category_name = path.basename(category)
        moves = glob.glob('{}/*'.format(category))
        first_move = path.basename(moves[0])
        target_color = first_move[:first_move.find('-')] # select from one color to avoid duplicate moves
        for move in moves:
            move = path.basename(move)
            dash_idx = move.find('-')
            color = move[:dash_idx]
            if color == target_color:
                move = move[dash_idx+1:-4] # exclude color and file extension
                move = '{}-{}-{}'.format(character_name, category_name, move)
                move_to_idx[move] = cur_idx
                idx_to_move[cur_idx] = move
                cur_idx += 1

save_dir = path.join(path.dirname(__file__), 'class_dicts')
makedirs(save_dir, exist_ok=True)

pickle.dump(move_to_idx, open(path.join(save_dir, 'move_to_idx.pkl'), 'wb'))
pickle.dump(idx_to_move, open(path.join(save_dir, 'idx_to_move.pkl'), 'wb'))
print(len(move_to_idx))