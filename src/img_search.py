import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
from lavis.models import load_model_and_preprocess
from PIL import Image, ImageDraw, ImageFont
import os

SPLIT = 'val'
BLIP2_MODEL = 'opt' # or 'opt'

dataset_path = Path('CIRCO')

with open(dataset_path / 'annotations' / f'{SPLIT}.json', "r") as f:
    annotations = json.load(f)

with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
    imgs_info = json.load(f)

img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                    imgs_info["images"]]
img_ids = [img_info["id"] for img_info in imgs_info["images"]]
img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}

current_path = os.getcwd()
absolute_path = os.path.abspath(current_path)

for ans in tqdm(annotations):
    ref_img_id = ans["reference_img_id"]
    gt_img_ids = ans["gt_img_ids"]

    rel_cap = ans["relative_caption"]
    shared_concept = ans["shared_concept"]
    blip2_caption = ans["blip2_caption"]
    gpt_35_turbo = ans["gpt-3.5-turbo"]
    gpt_35_turbo_none = ans["gpt-3.5-turbo_none"]

    id = ans["id"]
    image_id_list = [ref_img_id] + gt_img_ids

    images = []
    for img_id in image_id_list:
        reference_img_id = str(img_id)
        reference_img_path = img_paths[img_ids_indexes_map[reference_img_id]]
        raw_image = Image.open(reference_img_path).convert('RGB')
        images.append(raw_image)
    
    ans["image_path"] = absolute_path
    num_images = len(images)
    image_width = 400 * num_images
    image_height = 560
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    image_size = (400, 400)
    image_position = [(i * image_size[0], 0) for i in range(num_images)]

    for img, pos in zip(images, image_position):
        img = img.resize(image_size, Image.ANTIALIAS)
        image.paste(img, pos)

    text_lines = ['relative_caption: ' + rel_cap, 'shared_concept: ' + shared_concept, 'gpt-3.5_shared: ' + gpt_35_turbo_none, 'blip2_caption: ' + blip2_caption, 'gpt-3.5: ' + gpt_35_turbo]
    font_path = '/sda/home/qianshengsheng/yifei/workspace/SSL-Backdoor/poison-generation/fonts/FreeMonoBold.ttf' 
    font = ImageFont.truetype(font_path, 20)
    text_position = (10, 405)
    line_spacing = 30
    for line in text_lines:
        draw.text(text_position, line, fill=(0, 0, 0), font=font)
        text_position = (text_position[0], text_position[1] + line_spacing)

    image.save('images/{}.jpg'.format(id))
    ans["image_path"] = absolute_path + '/images/{}.jpg'.format(id)

with open("CIRCO/annotations/val_path.json", "w") as f:
    f.write(json.dumps(annotations, indent=4))