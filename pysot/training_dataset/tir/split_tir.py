from os.path import join
from os import mkdir
from os import listdir
from shutil import move
from shutil import rmtree
import glob

TIR_base_path = './TIRDataset-MMNet'
ann_base_path = join(TIR_base_path, 'Annotations/TIR/train/')
img_base_path = join(TIR_base_path, 'Data/TIR/train/')
sub_sets_original = sorted({'TIR_training_001', 'TIR_training_002', 'TIR_training_003'})
sub_sets = sorted({'a', 'b', 'c'})

for sub, sub_set in enumerate(sub_sets):
    sub_set_ann_path = join(ann_base_path, sub_set)
    sub_set_img_path = join(img_base_path, sub_set)
    sub_set_base_path = join(TIR_base_path, sub_sets_original[sub])
    videos = sorted(listdir(sub_set_base_path))
    for video in videos:
        video_base_path = join(sub_set_base_path, video)
        video_ann_path = join(sub_set_ann_path, video)
        video_img_path = join(sub_set_img_path, video)
        xml_list = sorted(glob.glob(join(video_base_path, '*.xml')))
        img_list = sorted(glob.glob(join(video_base_path, '*.jpg')))
        video_size = len(listdir(video_base_path))
        if video_size > 0 and len(xml_list) == len(img_list):
            for vi in range(video_size // 2):

                xml = xml_list[vi]
                img = img_list[vi]
                if xml.split('/')[-1].split('.')[0] == img.split('/')[-1].split('.')[0]:
                    try:
                        move(xml, join(video_ann_path, xml.split('/')[-1]))
                    except FileNotFoundError:
                        mkdir(video_ann_path)
                        move(xml, join(video_ann_path, xml.split('/')[-1]))

                    try:
                        move(img, join(video_img_path, img.split('/')[-1]))
                    except FileNotFoundError:
                        mkdir(video_img_path)
                        move(img, join(video_img_path, img.split('/')[-1]))
    rmtree(sub_set_base_path)