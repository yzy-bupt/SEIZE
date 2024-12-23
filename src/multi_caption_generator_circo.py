import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
from lavis.models import load_model_and_preprocess

SPLIT = 'test'
BLIP2_MODEL = 'opt' # 'opt' or 't5'
MULTI_CAPTION = True
NUM_CAPTION = 15

output_json = '{}_{}_multi.json'.format(SPLIT, BLIP2_MODEL) if MULTI_CAPTION else '{}_{}.json'.format(SPLIT, BLIP2_MODEL)
dataset_path = Path('CIRCO')

with open(dataset_path / 'annotations' / f'{SPLIT}_raw.json', "r") as f:
    annotations = json.load(f)

with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
    imgs_info = json.load(f)

img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                    imgs_info["images"]]
img_ids = [img_info["id"] for img_info in imgs_info["images"]]
img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}


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

for ans in tqdm(annotations):
    ref_img_id = ans["reference_img_id"]
    rel_cap = ans["relative_caption"]

    reference_img_id = str(ref_img_id)
    reference_img_path = img_paths[img_ids_indexes_map[reference_img_id]]

    raw_image = Image.open(reference_img_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    if MULTI_CAPTION:
        caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=NUM_CAPTION)
    else:
        caption = model.generate({"image": image})

    if MULTI_CAPTION:
        ans["multi_caption_{}".format(BLIP2_MODEL)] = caption
    else:
        ans["blip2_caption_{}".format(BLIP2_MODEL)] = caption[0]
    # with open("CIRCO/annotations/blip2_caption_t5.json", "a") as f:
    #     f.write(json.dumps(ans, indent=4) + '\n')

with open("CIRCO/annotations/{}".format(output_json), "w") as f:
    f.write(json.dumps(annotations, indent=4))