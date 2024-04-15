from argparse import ArgumentParser

class args_define():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", type=str, default='circo', choices=['cirr', 'circo', 'fashioniq'], help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", choices=['CIRR', 'CIRCO', 'FashionIQ'], default='CIRCO')
    parser.add_argument("--eval-type", type=str, choices=['ViT-B/32', 'ViT-L/14', 'ViT-H-14', 'ViT-g-14', 'ViT-bigG-14'], default='ViT-L/14', help="CLIP pretrained models")
    parser.add_argument("--type", type=str, default='L')

    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'], help="Preprocess pipeline to use")
    parser.add_argument("--phi-checkpoint-name", type=str)
    ### 'circo_L_gpt4_2.1_neg_0.13_pos'
    ### 'test_L_momentum_1.5_diff_pos_eql_zero' 'test_L_rank_5e-7' # 'test_L_m_2.1_neg_0.13_pos' 'cirr_G_m_1.0_neg_0.22_pos' 'fashioniq_L_searle_0.0_neg_0.0_pos'
    parser.add_argument("--submission-name", type=str, default='MM_only_pos_1000', help="Filename of the generated submission file") # 'multi_opt_gpt35_15_momentum_0.3' 'multi_opt_gpt35_15_debiased_0.1_sum'
    parser.add_argument("--caption_type", type=str, default='opt', choices=['none', 't5', 'opt'])
    parser.add_argument("--is_image_tokens", type=bool, default=False)
    parser.add_argument("--is_gpt_caption", type=bool, default=True)
    parser.add_argument("--is_rel_caption", type=bool, default=True)
    parser.add_argument("--multi_caption", type=bool, default=True)
    parser.add_argument("--nums_caption", type=int, default=15)
    
    parser.add_argument("--use_momentum_strategy", type=bool, default=True)
    parser.add_argument("--pos_factor", type=float, default=1000) # 5e-7 0.13 0.22 0.31 0.0005
    parser.add_argument("--neg_factor", type=float, default=0.0,) # 5e-7 2.1 1.0 5.0 0.25 
    parser.add_argument("--momentum_factor", type=float, default=0.13) # 5e-7

    ###
    parser.add_argument("--use_debiased_sample", type=bool, default=False)
    parser.add_argument("--debiased_temperature", type=float, default=0.01)


    ###
    parser.add_argument("--is_pre_features", type=bool, default=False)
    parser.add_argument("--is_pre_tokens", type=bool, default=False)
    
    parser.add_argument("--is_gpt_predicted_features", type=bool, default=False)
    parser.add_argument("--is_blip_predicted_features", type=bool, default=False)
    parser.add_argument("--is_features_save", type=bool, default=True)
    

    args = parser.parse_args()

    