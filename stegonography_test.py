import time
start_time = time.time()
from stegano import lsb
import os
from shutil import copyfile
from multiprocessing import Pool
from random_word import RandomWords

IMAGE_FOLDER = "downloads/init_test/"
DESTINATION_FOLDER_SECRET = "dataset/hidden"
DESTINATION_FOLDER_NORMAL = "dataset/open"
#LINK_FILE = "downloads/links.txt"

FLIP_BIT = False
LINE_TRACKER = 1
CORES = 40

FILES_TO_HIDE = []
FILES_TO_COPY = []

#file = open(LINK_FILE, 'r')
#Lines = file.readlines()

def arr_to_str(list):
    str = ""
    for word in list:
        str += word
        str += " "
    return str

def store_secret(file):
    string = ""
    while True:
        try:
            string = RandomWords().get_random_words(limit=10)
        except:
            continue
        break
    string = arr_to_str(string)
    print(string)
    secret = lsb.hide(os.path.join(IMAGE_FOLDER, file),string)
    secret.save(os.path.join(DESTINATION_FOLDER_SECRET,file))

def store_nothing(file):
    copyfile(os.path.join(IMAGE_FOLDER,file),os.path.join(DESTINATION_FOLDER_NORMAL,file))

# Sort files 50/50 with & without secret message:
for file in os.listdir(IMAGE_FOLDER):
    if FLIP_BIT:
        FILES_TO_HIDE.append(file)
        FLIP_BIT =  False
    else:
        FILES_TO_COPY.append(file)
        FLIP_BIT = True


p = Pool(CORES)
p.map(store_secret,FILES_TO_HIDE)
p.map(store_nothing,FILES_TO_COPY)

elapsed_time = time.time() - start_time
print("Total elapsed time: " + str(elapsed_time))
