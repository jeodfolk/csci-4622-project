from PIL import Image, ImageSequence
import glob
import os
import errno

# folders to convert
characters = [
                'ryu', 'oro', 'dudley', 'elena', 'hugo', 'ken',
                'yun', 'remy', 'q', 'chun-li', 'makoto', 'twelve', 'yang', 'gill'
                'akuma', 'urien', 'necro', 'ibuki', 'sean', 'alex'
             ]

for character in characters:
    paths = glob.glob("{}/*/*.gif".format(character))

    for path in paths:
        # get folder name for images to be saved under
        folder = os.path.relpath(os.path.splitext(path)[0], character)
        png_path = os.path.join("{}-png".format(character), folder)
        os.makedirs(png_path, exist_ok=True)
        img = Image.open(path)

        # Save all frames of gif as individual png in new folder
        frame_num = 0
        for frame in ImageSequence.Iterator(img):
            frame.save("{}/{}.png".format(png_path, frame_num))
            frame_num += 1