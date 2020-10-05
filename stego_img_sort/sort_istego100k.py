import json, os
from shutil import copyfile

json_path = "img_properties.json"
src_path = "E:\\ds-1\\stego\\"
dst_path = "E:\\ds-4\\"

with open(json_path, 'r', encoding='utf-8', errors='ignore') as the_file:
    params = json.load(the_file)
    for image in params:
        src_img = src_path + image
        dst_img = str(dst_path) + str(params[image]["steg_algorithm"]) + "\\" + str(image)
        copyfile(src_img,dst_img)
