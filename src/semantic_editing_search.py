import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import pickle
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
from utils import extract_image_features, device, collate_fn, extract_pseudo_tokens_with_phi, PROJECT_ROOT, targetpad_transform, encode_with_pseudo_tokens, Phi



pretraining = {
    'ViT-B-32':'laion2b_s34b_b79k',
    'ViT-B-16':'laion2b_s34b_b88k',
    'ViT-L-14':'laion2b_s32b_b82k',
    'ViT-H-14':'laion2b_s32b_b79k',
    'ViT-g-14':'laion2b_s34b_b88k',
    'ViT-bigG-14':'laion2b_s39b_b160k'
}


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str, clip_model_name: str, ref_names_list: List[str],
                                       pseudo_tokens: torch.Tensor, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    if clip_model_name in ['ViT-g-14', 'ViT-H-14', 'ViT-bigG-14']:
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretraining[clip_model_name])
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval()

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)
    # yzy
    if args.is_pre_features:
        index_features = torch.load(f'feature/{args.dataset}/{args.type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/index_names.npy')
        index_names = index_names.tolist()
    else:
        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names,
                                 ref_names_list, pseudo_tokens, args.nums_caption)

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

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], ref_names_list: List[str], pseudo_tokens: List[str], nums_caption) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    if args.is_gpt_predicted_features:
        reference_names = np.load(f'feature/{args.dataset}/reference_names.npy')
        pairs_id = np.load(f'feature/{args.dataset}/pairs_id.npy')
        group_members = np.load(f'feature/{args.dataset}/group_members.npy')
        predicted_features = torch.load(f'feature/{args.dataset}/{args.type}/gpt_predicted_features.pt')

        reference_names = reference_names.tolist()
        pairs_id = pairs_id.tolist()
        group_members = group_members.tolist()

    else:
        predicted_features, reference_names, pairs_id, group_members = \
            cirr_generate_test_predictions(clip_model, relative_test_dataset, ref_names_list, pseudo_tokens)
        
        np.save(f'feature/{args.dataset}/reference_names.npy', reference_names)
        np.save(f'feature/{args.dataset}/pairs_id.npy', pairs_id)
        np.save(f'feature/{args.dataset}/group_members.npy', group_members)
        torch.save(predicted_features, f'feature/{args.dataset}/{args.type}/gpt_predicted_features.pt')

    if args.use_momentum_strategy:
        if args.is_gpt_predicted_features:
            blip_predicted_features = torch.load(f'feature/{args.dataset}/{args.type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _, _, _ = \
                cirr_generate_test_predictions(clip_model, relative_test_dataset, ref_names_list, pseudo_tokens, True)
            
            torch.save(blip_predicted_features, f'feature/{args.dataset}/{args.type}/blip_predicted_features.pt')
    
    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()


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


def cirr_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset, ref_names_list: List[str],
                                   pseudo_tokens: torch.Tensor, use_momentum_strategy=False) -> \
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
    if args.type in ['G', 'H', 'bigG']:
        tokenizer = open_clip.get_tokenizer(args.eval_type)
    else:
        tokenizer = clip.tokenize

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
        if args.is_gpt_caption:
            if use_momentum_strategy:
                input_captions = multi_caption
            else:
                input_captions = multi_gpt_caption

        else:
            if args.is_rel_caption:
                input_captions = [f"a photo that {caption}" for caption in relative_captions]
            else:
                input_captions = multi_caption[0]


        batch_tokens = None# torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        
        if args.multi_caption:
            text_features_list = []
            for cap in input_captions:
                tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
                if args.type in ['G', 'H', 'bigG']:
                    text_features = clip_model.encode_text(tokenized_input_captions)
                else:
                    text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
                text_features_list.append(text_features)
            text_features_list = torch.stack(text_features_list)
            text_features = torch.mean(text_features_list, dim=0)

        else:
            tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
            if args.type in ['G', 'H', 'bigG']:
                text_features = clip_model.encode_text(tokenized_input_captions)
            else:
                text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)

        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, clip_model_name: str, ref_names_list: List[str],
                                        pseudo_tokens: torch.Tensor, preprocess: callable,
                                        submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    if clip_model_name in ['ViT-g-14', 'ViT-H-14', 'ViT-bigG-14']:
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretraining[clip_model_name])
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)

    # yzy
    if args.is_pre_features:
        index_features = torch.load(f'feature/{args.dataset}/{args.type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/index_names.npy')
        index_names = index_names.tolist()
    else:

        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)


    # for idx in range(15):
    #     # Get the predictions dict
    #     queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
    #                                                         index_names, ref_names_list, pseudo_tokens, idx + 1)

    #     submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    #     submissions_folder_path.mkdir(exist_ok=True, parents=True)

    #     with open(submissions_folder_path / f"{submission_name}_{idx + 1}.json", 'w+') as file:
    #         json.dump(queryid_to_retrieved_images, file, sort_keys=True)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
                                                           index_names, ref_names_list, pseudo_tokens, args.nums_caption)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRCODataset, ref_names_list: List[str],
                                    pseudo_tokens: torch.Tensor, use_momentum_strategy=False, debiased_id=-1) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    # yzy num_workers
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []
    if args.type in ['G', 'H', 'bigG']:
        tokenizer = open_clip.get_tokenizer(args.eval_type)
    else:
        tokenizer = clip.tokenize

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        blip2_caption = batch['blip2_caption_{}'.format(args.caption_type)]
        gpt_caption = batch['gpt_caption_{}'.format(args.caption_type)]
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        # yzy
        if args.is_image_tokens:
            if args.is_gpt_caption:
                if args.is_gpt_caption:
                    input_captions = [f"a photo of $ shows that {caption}" for caption in gpt_caption]
                else:
                    input_captions = [f"a photo of $ shows that {caption}" for caption in gpt_caption]
            else:
                if args.is_rel_caption:
                    if use_momentum_strategy:
                        input_captions = ["a photo of $" for caption in relative_captions]
                    else:
                        input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
                else:
                    input_captions = ["a photo of $" for caption in relative_captions]
        else:
            if args.is_gpt_caption:
                if args.multi_caption:
                    if use_momentum_strategy:
                        if debiased_id != -1:
                            input_captions = multi_caption[debiased_id]
                        else:
                            input_captions = multi_caption
                    else:
                        if debiased_id != -1:
                            input_captions = multi_gpt_caption[debiased_id]
                        else:
                            input_captions = multi_gpt_caption
                else:
                    if use_momentum_strategy:
                        input_captions = blip2_caption
                    else:
                        input_captions = [f"{caption}" for caption in gpt_caption]
            else:
                if args.multi_caption and args.is_rel_caption:
                    input_captions = multi_caption
                    for i in range(len(input_captions)): 
                        input_captions[i] = [f"a photo of {input_captions[i][inx]} that {relative_captions[inx]}" for inx in range(len(input_captions[i]))]
                else:
                    input_captions = [f"a photo that {caption}" for caption in relative_captions]

        if pseudo_tokens is not None:
            batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        else:
            batch_tokens = None
        if args.multi_caption and debiased_id == -1:
            text_features_list = []
            for cap in input_captions:
                tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
                if args.type in ['G', 'H', 'bigG']:
                    text_features = clip_model.encode_text(tokenized_input_captions)
                else:
                    text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
                text_features_list.append(text_features)
            text_features_list = torch.stack(text_features_list)
            text_features = torch.mean(text_features_list, dim=0)

        else:
            tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
            if args.type in ['G', 'H', 'bigG']:
                text_features = clip_model.encode_text(tokenized_input_captions)
            else:
                text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor, nums_caption) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    if args.is_gpt_predicted_features:
        if args.use_debiased_sample:
            predicted_features_list = []
            for i in range(nums_caption):
                predicted_features = torch.load('feature/debiased/gpt_predicted_features_{}.pt'.format(i))
                predicted_features_list.append(predicted_features)
            query_ids = np.load('feature/query_ids.npy')

        else:
            predicted_features = torch.load(f'feature/{args.dataset}/{args.type}/gpt_predicted_features.pt')
            query_ids = np.load(f'feature/{args.dataset}/{args.type}/query_ids.npy')
    else:
        if args.use_debiased_sample:
            predicted_features_list = []
            for i in range(nums_caption):
                predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset,
                                                                                ref_names_list, pseudo_tokens, debiased_id=i)
                torch.save(predicted_features, 'feature/debiased/gpt_predicted_features_{}.pt'.format(i))
                predicted_features_list.append(predicted_features)
        else:
            predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset,
                                                                            ref_names_list, pseudo_tokens)
            
            if args.is_features_save:
                np.save(f'feature/{args.dataset}/{args.type}/query_ids.npy', query_ids)
                torch.save(predicted_features, f'feature/{args.dataset}/{args.type}/gpt_predicted_features.pt')
    
    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    if args.use_momentum_strategy:
        if args.is_blip_predicted_features:
            if args.use_debiased_sample:
                blip_predicted_features_list = []
                for i in range(nums_caption):
                    blip_predicted_features = torch.load('feature/debiased/blip_predicted_features_{}.pt'.format(i))
                    blip_predicted_features_list.append(blip_predicted_features)
            else:
                blip_predicted_features = torch.load(f'feature/{args.dataset}/{args.type}/blip_predicted_features.pt')
        else:
            if args.use_debiased_sample:
                blip_predicted_features_list = []
                for i in range(nums_caption):
                    blip_predicted_features, _ = circo_generate_test_predictions(clip_model, relative_test_dataset,
                                                                                    ref_names_list, pseudo_tokens, True, debiased_id=i)
                    torch.save(blip_predicted_features, 'feature/debiased/blip_predicted_features_{}.pt'.format(i))
                    blip_predicted_features_list.append(blip_predicted_features)
            else:
                blip_predicted_features, _ = circo_generate_test_predictions(clip_model, relative_test_dataset,
                                                                                ref_names_list, pseudo_tokens, True)
                
                if args.is_features_save:
                    torch.save(blip_predicted_features, f'feature/{args.dataset}/{args.type}/blip_predicted_features.pt')
        
        ref_index = [index_names.index(ref_name) for ref_name in ref_names_list]
        if args.use_debiased_sample:
            neg_diff_val = []
            for i in range(nums_caption):
                gpt_features = predicted_features_list[i]
                blip_features = blip_predicted_features_list[i]
                # neg_diff_val.append(torch.sum(1 - (gpt_features * blip_features)).item())

                similarity_after = gpt_features @ index_features.T
                similarity_before = blip_features @ index_features.T
                diff = similarity_after - similarity_before
                diff[diff > 0] = 0
                diff = -diff
                diff = torch.topk(diff, dim=-1, k=50).values
                sum_diff = torch.sum(diff)
                # sum_diff = torch.sum(diff < 0)
                neg_diff_val.append(sum_diff.item())

            neg_diff_val_tensor = torch.tensor(neg_diff_val).float().to(device)
            print(neg_diff_val_tensor)
            debiased_weight = torch.softmax(neg_diff_val_tensor / torch.max(neg_diff_val_tensor) / args.debiased_temperature, 0)
            print(debiased_weight)
            predicted_features_tensor = torch.stack(predicted_features_list)
            if 0:
                debiased_features = torch.mean(predicted_features_tensor, dim=0)
            else:
                debiased_features = torch.sum(predicted_features_tensor * debiased_weight.unsqueeze(1).unsqueeze(2), dim=0)
            similarity = debiased_features @ index_features.T


        else:
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
    if clip_model_name in ['ViT-g-14', 'ViT-H-14', 'ViT-bigG-14']:
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretraining[clip_model_name])
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess)

    if args.is_pre_features:
        index_features = torch.load(f'feature/{args.dataset}/{args.type}/{dress_type}/index_features.pt')
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
    if args.is_gpt_predicted_features:
        target_names = np.load(f'feature/{args.dataset}/{dress_type}/target_names.npy')
        predicted_features = torch.load(f'feature/{args.dataset}/{args.type}/{dress_type}/gpt_predicted_features.pt')
        target_names = target_names.tolist()
    else:
        predicted_features, target_names = fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list,
                                                                        pseudo_tokens)
        np.save(f'feature/{args.dataset}/{dress_type}/target_names.npy', target_names)
        torch.save(predicted_features, f'feature/{args.dataset}/{args.type}/{dress_type}/gpt_predicted_features.pt')

    if args.use_momentum_strategy:
        if args.is_gpt_predicted_features:
            blip_predicted_features = torch.load(f'feature/{args.dataset}/{args.type}/{dress_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _ = \
                fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list, pseudo_tokens, True)
            
            torch.save(blip_predicted_features, f'feature/{args.dataset}/{args.type}/{dress_type}/blip_predicted_features.pt')

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
    if args.type in ['G', 'H', 'bigG']:
        tokenizer = open_clip.get_tokenizer(args.eval_type)
    else:
        tokenizer = clip.tokenize

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
        if pseudo_tokens is not None and args.is_image_tokens:
            batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        else:
            batch_tokens = None
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
            if args.type in ['G', 'H', 'bigG']:
                text_features = clip_model.encode_text(tokenized_input_captions)
            else:
                text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
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
    if args.eval_type == 'oti':
        experiment_path = PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset.lower() / 'test' / args.exp_name

        with open(experiment_path / 'hyperparameters.json') as f:
            hyperparameters = json.load(f)

        pseudo_tokens = torch.load(experiment_path / 'ema_oti_pseudo_tokens.pt', map_location=device)
        with open(experiment_path / 'image_names.pkl', 'rb') as f:
            ref_names_list = pickle.load(f)

        clip_model_name = hyperparameters['clip_model_name']
        clip_model, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)

        if args.preprocess_type == 'targetpad':
            # print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        elif args.preprocess_type == 'clip':
            # print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

    elif args.eval_type in ['phi', 'ViT-B/32', 'ViT-L/14', 'ViT-H-14', 'ViT-g-14', 'ViT-bigG-14']:
        if args.eval_type == 'phi':
            phi_path = PROJECT_ROOT / 'data' / "phi_models" / args.exp_name
            if not phi_path.exists():
                raise ValueError(f"Experiment {args.exp_name} not found")

            hyperparameters = json.load(open(phi_path / "hyperparameters.json"))
            clip_model_name = hyperparameters['clip_model_name']
            clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

            phi = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                      output_dim=clip_model.token_embedding.embedding_dim, dropout=hyperparameters['phi_dropout']).to(
                device)

            phi.load_state_dict(
                torch.load(phi_path / 'checkpoints' / args.phi_checkpoint_name, map_location=device)[
                    phi.__class__.__name__])
            phi = phi.eval()

        else: 
            clip_model_name = args.eval_type

            if args.is_image_tokens:
                phi, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                                        backbone=clip_model_name)

                phi = phi.to(device).eval()
            else:
                phi = None

            if clip_model_name in ['ViT-g-14', 'ViT-H-14', 'ViT-bigG-14']:
                clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretraining[clip_model_name])
            else:
                clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, 224)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

        if args.dataset.lower() == 'fashioniq':
            relative_test_dataset = FashionIQDataset(args.dataset_path, 'val', ['dress', 'toptee', 'shirt'],
                                                    'relative', preprocess, no_duplicates=True)
        elif args.dataset.lower() == 'cirr':
            relative_test_dataset = CIRRDataset(args.dataset_path, 'test', 'relative', preprocess, no_duplicates=True)
        elif args.dataset.lower() == 'circo':
            relative_test_dataset = CIRCODataset(args.dataset_path, 'test', 'relative', preprocess)
        else:
            raise ValueError("Dataset not supported")

        clip_model = clip_model.float().to(device)
        if args.is_pre_tokens:
            if phi:
                pseudo_tokens = torch.load(f'feature/{args.dataset}/{args.type}/predicted_tokens.pt')
                pseudo_tokens = pseudo_tokens.to(device)
            else:
                pseudo_tokens = None
            ref_names_list = np.load(f'feature/{args.dataset}/names_list.npy')
            ref_names_list = ref_names_list.tolist()
        else:
            pseudo_tokens, ref_names_list = extract_pseudo_tokens_with_phi(clip_model, phi, relative_test_dataset)

    else:
        raise ValueError("Eval type not supported")

    # print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model_name, ref_names_list, pseudo_tokens,
                                           preprocess, args.submission_name)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path, clip_model_name, ref_names_list, pseudo_tokens,
                                            preprocess, args.submission_name)
    elif args.dataset.lower() == 'fashioniq':
        recalls_at10 = []
        recalls_at50 = []
        for dress_type in ['shirt', 'dress', 'toptee']: # , 'shirt', 'toptee'
            fiq_metrics = fiq_val_retrieval(args.dataset_path, dress_type, clip_model_name, ref_names_list,
                                            pseudo_tokens, preprocess)
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
