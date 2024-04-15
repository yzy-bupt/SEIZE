import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
from lavis.models import load_model_and_preprocess

SPLIT = 'val'
# DRESS = 'toptee' # 'shirt' 'toptee'
BLIP2_MODEL = 'opt' # or 'opt'
MULTI_CAPTION = True
NUM_CAPTION = 15

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

if BLIP2_MODEL == 'opt':
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )
else:
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )
# model = model.float()
    
for DRESS in ['dress', 'shirt', 'toptee']:
    output_json = '{}_{}_multi.json'.format(SPLIT, BLIP2_MODEL) if MULTI_CAPTION else '{}_{}.json'.format(SPLIT, BLIP2_MODEL)
    dataset_path = Path('/sda/home/qianshengsheng/yzy/FashionIQ')

    with open(dataset_path / 'captions' / f'cap.{DRESS}.{SPLIT}.json') as f:
        annotations = json.load(f)

    # with open(dataset_path / 'image_splits' / f'split.{DRESS}.{SPLIT}.json') as f:
    #     name_to_relpath = json.load(f)

    # cnt = 0
    # NUM_QUERY = 200
    # new_annotations = []

    for ans in tqdm(annotations):
        # cnt += 1
        # if cnt > NUM_QUERY:
        #     break
        ref_img_name = ans["candidate"] + '.png'

        reference_img_path = dataset_path / 'images' / ref_img_name

        raw_image = Image.open(reference_img_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        if MULTI_CAPTION:
            caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=NUM_CAPTION)
        else:
            caption = model.generate({"image": image})
        # print(caption)

        # raw_image.save('aaa.png')

        if MULTI_CAPTION:
            ans["multi_caption_{}".format(BLIP2_MODEL)] = caption
        else:
            ans["blip2_caption_{}".format(BLIP2_MODEL)] = caption[0]

        # new_annotations.append(ans)
        # with open("CIRCO/annotations/blip2_caption_t5.json", "a") as f:
        #     f.write(json.dumps(ans, indent=4) + '\n')

    with open(dataset_path / f'cap.{DRESS}.{SPLIT}.json', "w") as f:
        f.write(json.dumps(annotations, indent=4))
