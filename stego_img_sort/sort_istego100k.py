import json, os, time
from shutil import copyfile
from multiprocessing import Pool

json_path = "img_properties.json"
src_path = "E:\\ds-1\\cover\\"
dst_path = "E:\\clean_sort\\"


def img_cpy(list):
    global src_path
    global dst_path
    src_img = src_path + list[0]
    dst_img = str(dst_path) + str(list[1]) + "\\" + str(list[0])
    copyfile(src_img,dst_img)

t = time.time()

if __name__ == '__main__':
    p = Pool(8)
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as the_file:
        params = json.load(the_file)
        images = []
        for image in params:
            images.append((image,params[image]["steg_algorithm"]))
        print(images)
        p.map(img_cpy, images)
    elapsed = time.time() - t
    print("Total time: " + str(elapsed))
