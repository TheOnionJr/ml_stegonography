from image_slicer import slice
import os
from multiprocessing import Pool
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

path = "F:\\sander\\unprocessed_dataset"
slices = 4
global save_path
save_path = "F:\\sander\\processed_dataset"
size = 1024,1024

global counter
counter = 0
count_max = 0

process_list = []

def counting(path):
    tmp_count = 0
    for pic in os.listdir(path):
        process_list.append(os.path.join(path, pic))
        tmp_count = tmp_count + 1
    return tmp_count

def read_dir(path):
    list = []
    for pic in os.listdir(path):
        list.append(os.path.join(path, pic))
    return list

def slice_func(file):
    try:
        slice(file, slices)
    except:
        print("Can't slice " + file + " deleting")
        os.remove(file)

def resize(filepath):
    size = (1024,1024)
    img = Image.open(filepath)
    img = img.resize((1024,1024))
    global counter
    counter = counter + 1
    name = str(counter) + ".jpg"
    img.save(os.path.join(save_path,name))
    os.remove(filepath)


if __name__ == '__main__':
    p = Pool(8)
    print("Detected " + counting(path) " pictures. Starting slicing...")
    p.map(slice_func, read_dir(path))
    print("Slicing completed. Resizing " + counting(path) + " images...")
    p.map(resize, read_dir(path))
    print("Script complete")
