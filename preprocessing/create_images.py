from PIL import Image
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.androconf import show_logging
import logging
import glob
import re
import os
import pickle
import string
import math
from tqdm import tqdm
import numpy as np
import optparse

parser = optparse.OptionParser()

parser.add_option('-s', '--source-dataset',
    action="store", dest="dataset_dir",
    help="Directory of the dataset to process", default="../dataset/")
parser.add_option('-d', '--img-dataset-dest',
    action="store", dest="dataset_dest",
    help="Destination of the images dataset that will be created", default="../dataset_img/")
parser.add_option('-t', '--dataset-type',
    action="store", dest="dataset_type",
    help="Type of dataset to build (bw, rgb)", default="rgb")

options, args = parser.parse_args()


show_logging(level=logging.ERROR)
logging.basicConfig(level=logging.ERROR)



root_dir = options.dataset_dir
images_root_dir = options.dataset_dest


color_map = Image.open('./map-saturation.png').load()


def parse_apk(path):
    """
    Parse an apk file to my custom bytecode output
    :param path: the path to the
    :rtype: string
    """
    # Load our example APK
    a = APK(path)
    # Create DalvikVMFormat Object
    #d = DalvikVMFormat(a)
    return a


def prepare_image(path):
    apk = parse_apk(path)
    data = apk.get_dex()
    size = math.ceil(math.sqrt(len(data)))
    data += bytes('\x00' * (size ** 2 - len(data)), 'utf-8')

    img = Image.frombytes('1', (size,size), apk.get_dex())

    return img #img.show()


def prepare_rgb_image(path):
    apk = parse_apk(path)
    data = apk.get_dex()
    data_length = len(data) - 1  # conto le coppie di bytes
    size = math.ceil(math.sqrt(data_length))
    rgb_data = np.empty(shape=(size,size,3), dtype='uint8')

    i = 0
    for row, col in zip(data, data[1:]):
        rgb_data[int(i/size), i%size] = np.array(color_map[row,col])
        i += 1

    while i < size ^ 2:
        rgb_data[int(i/size), i%size] = [255,255,255]
        i += 1

    img = Image.fromarray(rgb_data)

    return img #img.show()


def prepare_img_dataset(type='bw'):
    """
    type: 'bw' or 'rgb' with color map
    """

    total = 0
    for _ in glob.iglob(root_dir + '**/*.apk', recursive=True):
        total += 1

    already_done = []
    for apk_file in glob.iglob(images_root_dir + '**/*.jpeg', recursive=True):
        m = re.match(images_root_dir + "(.*)\/(.+).jpeg", apk_file)
        path = m.group(1)
        filename = m.group(2)
        already_done.append(path + filename)


    with tqdm(total=total) as pbar:
        for apk_file in glob.iglob(root_dir + '**/*.apk', recursive=True):
            try:
                m = re.match(root_dir + "(.*)\/(.+).apk", apk_file)
                path = m.group(1)
                filename = m.group(2)
                if path + filename not in already_done:
                    if not os.path.exists(images_root_dir + path):
                        os.makedirs(images_root_dir + path)
                    if type == 'bw':
                        img = prepare_image(apk_file)
                    elif type == 'rgb':
                        img = prepare_rgb_image(apk_file)
                    img.save(images_root_dir + path + "/" + filename + ".jpeg", "JPEG")
            except Exception as e:
                logging.error('Failed: ' + apk_file + "\t" + str(e))
            pbar.update(1)


def get_max_size(): #2640
    max_size = 0
    for apk_file in glob.iglob(images_root_dir + '**/*.jpeg', recursive=True):
        img = Image.open(apk_file)
        size = img.size[0]
        if size > max_size:
            max_size = size
    return max_size


def reformat_image(imageFilePath, type):
    '''
    Resize image to 2640x2640px adding white backgroud
    '''
    max_size = 2640
    image = Image.open(imageFilePath, 'r')
    size = image.size[0]
    margin = int((max_size - size) / 2)

    if type == 'bw':
        background = Image.new('1', (max_size, max_size), 1)
    elif type == 'rgb':
        background = Image.new('RGB', (max_size, max_size), (255,255,255))
    background.paste(image, box=(margin, margin))
    background.save(imageFilePath)


def reformat_all_images(type='bw'):
    total = 0
    for _ in glob.iglob(images_root_dir + '**/*.jpeg', recursive=True):
        total += 1

    with tqdm(total=total) as pbar:
        for img_file in glob.iglob(images_root_dir + '**/*.jpeg', recursive=True):
            reformat_image(img_file, type)
            pbar.update(1)


if __name__ == '__main__':
    prepare_img_dataset(options.dataset_type)
    reformat_all_images(options.dataset_type)
