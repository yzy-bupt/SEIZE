import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
from args import args_define
from typing import List, Tuple, Dict

import clip
import open_clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIRRDataset, CIRCODataset, FashionIQDataset
from utils import extract_image_features, device, collate_fn, PROJECT_ROOT, targetpad_transform


name2model = {
    'SEIZE-B':'ViT-B-32',
    'SEIZE-L':'ViT-L-14',
    'SEIZE-H':'ViT-H-14',
    'SEIZE-g':'ViT-g-14',
    'SEIZE-G':'ViT-bigG-14',
    'SEIZE-CoCa-B':'coca_ViT-B-32',
    'SEIZE-CoCa-L':'coca_ViT-L-14'
}

pretrained = {
    'ViT-B-32':'openai',
    'ViT-L-14':'openai', # For fair comparison, previous work used opanai's CLIP instead of open_clip
    'ViT-H-14':'laion2b_s32b_b79k', # Models larger than ViT-H only on open_clip, using laion2b uniformly
    'ViT-g-14':'laion2b_s34b_b88k',
    'ViT-bigG-14':'laion2b_s39b_b160k',
    'coca_ViT-B-32':'mscoco_finetuned_laion2b_s13b_b90k', # 'laion2b_s13b_b90k'
    'coca_ViT-L-14':'mscoco_finetuned_laion2b_s13b_b90k'  # 'laion2b_s13b_b90k'
}


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretrained[clip_model_name])
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)

    if os.path.exists(f'feature/{args.dataset}/{args.model_type}/index_features.pt'):
        index_features = torch.load(f'feature/{args.dataset}/{args.model_type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/index_names.npy')
        index_names = index_names.tolist()
    else:
        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'cirr'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{args.model_type}_{args.gpt_version}_{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{args.model_type}_{args.gpt_version}_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    feat_dataset_path = f'feature/{args.dataset}'
    if os.path.exists(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt'):
        reference_names = np.load(f'{feat_dataset_path}/reference_names.npy')
        pairs_id = np.load(f'{feat_dataset_path}/pairs_id.npy')
        group_members = np.load(f'{feat_dataset_path}/group_members.npy')
        predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
        reference_names = reference_names.tolist()
        pairs_id = pairs_id.tolist()
        group_members = group_members.tolist()

    else:
        predicted_features, reference_names, pairs_id, group_members = \
            cirr_generate_test_predictions(clip_model, relative_test_dataset)      
        np.save(f'{feat_dataset_path}/reference_names.npy', reference_names)
        np.save(f'{feat_dataset_path}/pairs_id.npy', pairs_id)
        np.save(f'{feat_dataset_path}/group_members.npy', group_members)
        torch.save(predicted_features, f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')

    if args.use_momentum_strategy:
        if os.path.exists(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt'):
            blip_predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _, _, _ = \
                cirr_generate_test_predictions(clip_model, relative_test_dataset, True)
            
            torch.save(blip_predicted_features, f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
    
    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    similarity_after = predicted_features @ index_features.T
    similarity_before = blip_predicted_features @ index_features.T

    diff_pos = similarity_after - similarity_before

    diff_pos[diff_pos < 0] = 0

    diff_neg = similarity_after - similarity_before

    diff_neg[diff_neg > 0] = 0

    similarity = similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos

    # similarity = similarity_after + args.momentum_factor * diff_neg + 0.3 * diff_pos

    # sorted_indices_before = torch.topk(similarity_before, dim=-1, k=similarity_before.shape[-1]).indices
    # sorted_indices_after = torch.topk(similarity_after, dim=-1, k=similarity_after.shape[-1]).indices

    # sorted_idx_before = torch.topk(sorted_indices_before, dim=-1, k=similarity_before.shape[-1], largest=False).indices
    # sorted_idx_after = torch.topk(sorted_indices_after, dim=-1, k=similarity_after.shape[-1], largest=False).indices

    # diff_neg = sorted_idx_before - sorted_idx_after
    # diff_neg[diff_neg > 0] = 0 

    # similarity = similarity_after + args.momentum_factor * diff_neg

    # for i in range(similarity_before.shape[0]):
    #     similarity[i][indexs[i]] = -1


    # Compute the distances and sort the results
    distances = 1 - similarity
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset, use_momentum_strategy=False) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []
    tokenizer = open_clip.get_tokenizer(name2model[args.model_type])

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        group_members = np.array(group_members).T.tolist()

        # input_captions = [
        #     f"a photo of $ that {rel_caption}" for rel_caption in relative_captions]
        if use_momentum_strategy:
            input_captions = multi_caption
        else:
            input_captions = multi_gpt_caption
    
        text_features_list = []
        for cap in input_captions:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            text_features_list.append(text_features)
        text_features_list = torch.stack(text_features_list)
        text_features = torch.mean(text_features_list, dim=0)

        predicted_features = F.normalize(text_features)
        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable,
                                        submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretrained[clip_model_name])
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)

    if os.path.exists(f'feature/{args.dataset}/{args.model_type}/index_features.pt'):
        index_features = torch.load(f'feature/{args.dataset}/{args.model_type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/index_names.npy')
        index_names = index_names.tolist()
    else:
        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
                                                           index_names)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{args.model_type}_{args.gpt_version}_{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRCODataset, use_momentum_strategy=False) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    # yzy num_workers
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []
    tokenizer = open_clip.get_tokenizer(name2model[args.model_type])

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        blip2_caption = batch['blip2_caption_{}'.format(args.caption_type)]
        gpt_caption = batch['gpt_caption_{}'.format(args.caption_type)]
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        if use_momentum_strategy:
            input_captions = multi_caption
        else:
            input_captions = multi_gpt_caption

        text_features_list = []
        for cap in input_captions:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            text_features_list.append(text_features)
        text_features_list = torch.stack(text_features_list)
        text_features = torch.mean(text_features_list, dim=0)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str]) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    feat_dataset_path = f'feature/{args.dataset}'
    if os.path.exists(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt'):
        predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
        query_ids = np.load(f'{feat_dataset_path}/query_ids.npy')
    else:
        predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset)
        np.save(f'{feat_dataset_path}/query_ids.npy', query_ids)
        torch.save(predicted_features, f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    if args.use_momentum_strategy:
        if os.path.exists(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt'):
            blip_predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _ = circo_generate_test_predictions(clip_model, relative_test_dataset, True)
            torch.save(blip_predicted_features, f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
        
        ref_names_list = np.load(f'{feat_dataset_path}/ref_names.npy')
        ref_names_list = ref_names_list.tolist()
        index_name_dict = {name: index for index, name in enumerate(index_names)}
        indexs = [index_name_dict[name] for name in ref_names_list]

        similarity_after = predicted_features @ index_features.T
        similarity_before = blip_predicted_features @ index_features.T

        diff_pos = similarity_after - similarity_before

        diff_pos[diff_pos < 0] = 0

        diff_neg = similarity_after - similarity_before

        diff_neg[diff_neg > 0] = 0

        similarity = similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos

        # similarity = similarity_after + args.momentum_factor * diff_neg + 0.3 * diff_pos

        # sorted_indices_before = torch.topk(similarity_before, dim=-1, k=similarity_before.shape[-1]).indices
        # sorted_indices_after = torch.topk(similarity_after, dim=-1, k=similarity_after.shape[-1]).indices

        # sorted_idx_before = torch.topk(sorted_indices_before, dim=-1, k=similarity_before.shape[-1], largest=False).indices
        # sorted_idx_after = torch.topk(sorted_indices_after, dim=-1, k=similarity_after.shape[-1], largest=False).indices

        # diff_neg = sorted_idx_before - sorted_idx_after
        # diff_neg[diff_neg > 0] = 0 

        # similarity = similarity_after + args.momentum_factor * diff_neg

        for i in range(similarity_before.shape[0]):
            similarity[i][indexs[i]] = -1

        # forward_steps = torch.zeros_like(similarity_after)
        # for i in range(sorted_indices_before.shape[0]):
        #     rank_blip = sorted_indices_before[i].numpy()
        #     rank_gpt = sorted_indices_after[i].numpy()
        #     idxs = np.arange(len(rank_blip))

        #     # 转换第一个向量为字典
        #     first_dict = {id: index for index, id in enumerate(rank_blip)}

        #     # 遍历第二个向量
        #     for idx, id in enumerate(rank_gpt):
        #         forward_steps[i][id] = first_dict[id] - idx
            
        # similarity = similarity_after + args.momentum_factor * forward_steps


    # Compute the similarity
    else:
        similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


@torch.no_grad()
def fiq_val_retrieval(dataset_path: str, dress_type: str, clip_model_name: str, ref_names_list: List[str],
                      pseudo_tokens: torch.Tensor, preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretrained[clip_model_name])
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess)

    if os.path.exists(f'feature/{args.dataset}/{dress_type}/{args.model_type}/index_features.pt'):
        index_features = torch.load(f'feature/{args.dataset}/{dress_type}/{args.model_type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/{dress_type}/index_names.npy')
        index_names = index_names.tolist()
    else:
        index_features, index_names = extract_image_features(classic_val_dataset, clip_model, dress_type=dress_type)

    # Define the relative dataset
    relative_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess)

    return fiq_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names, ref_names_list,
                                   pseudo_tokens)


@torch.no_grad()
def fiq_compute_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.Tensor,
                            index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
    """

    dress_type = relative_val_dataset.dress_types[0]
    # Generate the predicted features
    feat_dataset_dress_path = f'feature/{args.dataset}/{dress_type}'
    if os.path.exists(f'{feat_dataset_dress_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt'):
        target_names = np.load(f'{feat_dataset_dress_path}/target_names.npy')
        predicted_features = torch.load(f'{feat_dataset_dress_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
        target_names = target_names.tolist()
    else:
        predicted_features, target_names = fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list,
                                                                        pseudo_tokens)
        np.save(f'{feat_dataset_dress_path}/target_names.npy', target_names)
        torch.save(predicted_features, f'{feat_dataset_dress_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')

    if args.use_momentum_strategy:
        if os.path.exists(f'{feat_dataset_dress_path}/{args.model_type}/blip_predicted_features.pt'):
            blip_predicted_features = torch.load(f'{feat_dataset_dress_path}/{args.model_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _ = \
                fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list, pseudo_tokens, True)
            
            torch.save(blip_predicted_features, f'{feat_dataset_dress_path}/{args.model_type}/blip_predicted_features.pt')

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    index_name_dict = {name: index for index, name in enumerate(index_names)}
    indexs = [index_name_dict[name] if name in index_name_dict else -1 for name in ref_names_list]


    similarity_after = predicted_features @ index_features.T
    similarity_before = blip_predicted_features @ index_features.T

    diff_pos = similarity_after - similarity_before

    diff_pos[diff_pos < 0] = 0

    diff_neg = similarity_after - similarity_before

    diff_neg[diff_neg > 0] = 0

    similarity = similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos

    for i in range(similarity_before.shape[0]):
        if indexs[i] != -1:
            similarity[i][indexs[i]] = -1

    # Compute the distances
    distances = 1 - similarity
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return {'fiq_recall_at10': recall_at10,
            'fiq_recall_at50': recall_at50}


@torch.no_grad()
def fiq_generate_val_predictions(clip_model: CLIP, relative_val_dataset: FashionIQDataset, ref_names_list: List[str],
                                 pseudo_tokens: torch.Tensor, use_momentum_strategy=False) -> Tuple[torch.Tensor, List[str]]:
    """
    Generates features predictions for the validation set of Fashion IQ.
    """

    # Create data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=16,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    tokenizer = open_clip.get_tokenizer(name2model[args.model_type])

    # Compute features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_captions']
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        # flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        # input_captions = [
        #     f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
        #     i in range(0, len(flattened_captions), 2)]
        # input_captions_reversed = [
        #     f"{flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}" for
        #     i in range(0, len(flattened_captions), 2)]
        
        if use_momentum_strategy:
            input_captions = multi_caption
        else:
            input_captions = multi_gpt_caption

        # input_captions = [
        #     f"a photo of $ that {in_cap}" for in_cap in input_captions]
        # tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
        # text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)

        # input_captions_reversed = [
        #     f"a photo of $ that {in_cap}" for in_cap in input_captions_reversed]
        # tokenized_input_captions_reversed = tokenizer(input_captions_reversed, context_length=77).to(device)
        # text_features_reversed = encode_with_pseudo_tokens(clip_model, tokenized_input_captions_reversed,
        #                                                    batch_tokens)


        text_features_list = []
        for cap in input_captions:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            text_features_list.append(text_features)
        text_features_list = torch.stack(text_features_list)
        text_features = torch.mean(text_features_list, dim=0)

        predicted_features = F.normalize(text_features)
        # predicted_features = F.normalize((F.normalize(text_features) + F.normalize(text_features_reversed)) / 2)
        # predicted_features = F.normalize((text_features + text_features_reversed) / 2)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, target_names_list


args = args_define.args

def main():
    if args.model_type in ['SEIZE-B', 'SEIZE-L', 'SEIZE-H', 'SEIZE-g', 'SEIZE-G', 'SEIZE-CoCa-B', 'SEIZE-CoCa-L']:
        clip_model_name = name2model[args.model_type]
        preprocess = targetpad_transform(1.25, 224)
    else:
        raise ValueError("Model type not supported")

    folder_path = f'feature/{args.dataset}/{args.model_type}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)
    elif args.dataset.lower() == 'fashioniq':
        recalls_at10 = []
        recalls_at50 = []
        for dress_type in ['shirt', 'dress', 'toptee']:
            fiq_metrics = fiq_val_retrieval(args.dataset_path, dress_type, clip_model_name, preprocess)
            recalls_at10.append(fiq_metrics['fiq_recall_at10'])
            recalls_at50.append(fiq_metrics['fiq_recall_at50'])

            for k, v in fiq_metrics.items():
                print(f"{dress_type}_{k} = {v:.2f}")
            print("\n")

        print(f"average_fiq_recall_at10 = {np.mean(recalls_at10):.2f}")
        print(f"average_fiq_recall_at50 = {np.mean(recalls_at50):.2f}")    
    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
