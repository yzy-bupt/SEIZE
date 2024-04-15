import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
from lavis.models import load_model_and_preprocess
from PIL import Image, ImageDraw, ImageFont
import os

SPLIT = 'test'

dataset_path = Path('CIRCO')

with open(dataset_path / 'annotations' / f'{SPLIT}.json', "r") as f:
    annotations = json.load(f)

with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
    imgs_info = json.load(f)

json_names = ['only_image_token', 'only_relative_text', 'SEARLE', 'SEARLE_t5_gpt35', 'SEARLE_opt_gpt35', 'opt_gpt35', 'none_gpt35', 'test_G_circo_gpt4']

tests = []
for js_name in json_names:
    with open("data/test_submissions/circo/" + "{}.json".format(js_name), "r") as f:
        tests.append(json.load(f))

img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                    imgs_info["images"]]
img_ids = [img_info["id"] for img_info in imgs_info["images"]]
img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}

current_path = os.getcwd()
absolute_path = os.path.abspath(current_path)
num_images = 5

cnt = 0
for ans in tqdm(annotations):
    cnt += 1
    if cnt <50:
        continue
    if cnt > 200:
        break
    ref_img_id = ans["reference_img_id"]
    id = str(ans["id"])

    # gt_img_ids = ans["gt_img_ids"]

    rel_cap = ans["relative_caption"]
    shared_concept = ans["shared_concept"]
    blip2_caption = ans["blip2_caption"]
    gpt_35_turbo = ans["gpt-3.5-turbo"]
    blip2_caption_t5 = ans["blip2_caption_t5"]
    gpt_35_turbo_t5 = ans["gpt-3.5-turbo_t5"]
    gpt_35_turbo_none = ans["gpt-3.5-turbo_none"]
    
    json_names = ['only_image_token', 'only_relative_text: '+rel_cap, 'SEARLE: $ + '+rel_cap, 'SEARLE_t5_gpt35: $ + '+gpt_35_turbo_t5, 'SEARLE_opt_gpt35: $ + '+gpt_35_turbo, 'opt_gpt35: '+gpt_35_turbo, 'SEARLE_none_gpt35: $ + '+gpt_35_turbo_none, 'none_gpt35: '+gpt_35_turbo_none]


    images = []
    for i in range(len(tests)):
        id = str(ans["id"])
        image_id_list = [str(ref_img_id)] + tests[i][id][:num_images - 1]
        for img_id in image_id_list:
            reference_img_path = img_paths[img_ids_indexes_map[img_id]]
            raw_image = Image.open(reference_img_path).convert('RGB')
            images.append(raw_image)
    
    image_width = 400 * num_images
    image_height = 220 + 440 * len(tests)
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    image_size = (400, 400)
    image_position = [((i % num_images) * image_size[0], 220 + (i // num_images) * (image_size[1] + 40)) for i in range(len(images))]

    for img, pos in zip(images, image_position):
        img = img.resize(image_size, Image.ANTIALIAS)
        image.paste(img, pos)

    text_lines = ['relative_caption: ' + rel_cap, 'shared_concept: ' + shared_concept, 'gpt-3.5_shared: ' + gpt_35_turbo_none, 'blip2_caption: ' + blip2_caption, 'gpt-3.5: ' + gpt_35_turbo, 'blip2_caption_t5: ' + blip2_caption_t5, 'gpt-3.5_t5:: ' + gpt_35_turbo_t5]
    font_path = '/sda/home/qianshengsheng/yifei/workspace/SSL-Backdoor/poison-generation/fonts/FreeMonoBold.ttf'  
    font = ImageFont.truetype(font_path, 20)
    text_position = (10, 5)
    line_spacing = 30
    for line in text_lines:
        draw.text(text_position, line, fill=(0, 0, 0), font=font)
        text_position = (text_position[0], text_position[1] + line_spacing)

    line_spacing = 440
    text_position = (10, 625)
    for line in json_names:
        draw.text(text_position, line, fill=(0, 0, 0), font=font)
        text_position = (text_position[0], text_position[1] + line_spacing)

    image.save('retrieval/{}.jpg'.format(id))