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

FLIP_BIT = False #Creating a 50/50 split between fals and positive cases
CORES = 40  #Amount of system cores to use

FILES_TO_HIDE = [] #List of files to insert LSB stegonography into
FILES_TO_COPY = []  #List of files to not have any stegonography in them

def arr_to_str(list): #Create a string from the word list
    str = ""
    for word in list:
        str += word
        str += " "
    return str

def store_secret(file):
    string = ""  #String to insert
    while True:  #Loop so that API dont time out
        try:
            string = RandomWords().get_random_words(limit=10) #Get 10 random words and put them in string
        except:
            continue
        break
    string = arr_to_str(string) #Cast from list to string
    print(string)   #Debug
    secret = lsb.hide(os.path.join(IMAGE_FOLDER, file),string) #Hide string in picture with LSB
    secret.save(os.path.join(DESTINATION_FOLDER_SECRET,file)) #Save new picture

def store_nothing(file):
    copyfile(os.path.join(IMAGE_FOLDER,file),os.path.join(DESTINATION_FOLDER_NORMAL,file)) #Filecopy of file without stegonography

# Sort files 50/50 with & without secret message:
for file in os.listdir(IMAGE_FOLDER):
    if FLIP_BIT:    #With secret message
        FILES_TO_HIDE.append(file)
        FLIP_BIT =  False
    else:           #Without
        FILES_TO_COPY.append(file)
        FLIP_BIT = True


p = Pool(CORES) #Create work pool
p.map(store_secret,FILES_TO_HIDE) #Start stegonography
p.map(store_nothing,FILES_TO_COPY)  #Copy files

elapsed_time = time.time() - start_time #check time passed for performance metric
print("Total elapsed time: " + str(elapsed_time)) #Print time passed
