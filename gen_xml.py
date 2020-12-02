import xml.etree.ElementTree as ET
import cv2
import glob
import os
import numpy as np
from crop_img import crop
from xml_to_csv import convert

characters = [
                'ryu', 'oro', 'ken',
                'yun', 'remy', 'q', 'makoto', 'twelve', 'yang',
                'urien', 'necro', 'ibuki', 'sean'
             ]

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


character = "zzzzz"
i = 0

save_path = "C:/Users/joe/Documents/Visual Code/code/4622/tracking/models/research/object_detection/data/ryu/"
for path in glob.iglob('**/*.png', recursive=True):
    if path.startswith(character) == False:
        for char in characters:
            if path.startswith(char):
                print("{} started".format(char))
                character = char
                i = 0
                break
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    x_min, y_min, x_max, y_max = crop(path)
    height = img.shape[0]
    width = img.shape[1]
    filename = character + str(i)

    GenerateXML(filename, save_path, width, height, character, save_path, x_min, y_min, x_max, y_max)
    cv2.imwrite(save_path + filename + ".png", img)
    
    #also save files in test or train with 20/80 split
    data_split = np.random.choice([True, False], p=[.20, .80])
    if data_split:
        GenerateXML(filename, save_path+"test/", width, height, character, save_path+"test/", x_min, y_min, x_max, y_max)
        cv2.imwrite(save_path+"test/" + filename + ".png", img) 
    else:
        GenerateXML(filename, save_path+"train/", width, height, character, save_path+"train/", x_min, y_min, x_max, y_max)
        cv2.imwrite(save_path+"train/" + filename + ".png", img) 
    i += 1
convert()