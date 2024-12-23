from argparse import ArgumentParser

class args_define():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='circo', choices=['cirr', 'circo', 'fashioniq'], help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", choices=['CIRR', 'CIRCO', 'FashionIQ'], default='CIRCO')
    parser.add_argument("--model_type", type=str, choices=['SEIZE-B', 'SEIZE-L', 'SEIZE-H', 'SEIZE-g', 'SEIZE-G', 'SEIZE-CoCa-B', 'SEIZE-CoCa-L'], default='SEIZE-G',
                        help="if 'SEIZE-B' uses the pre-trained CLIP ViT-B/32,"
                             "if 'SEIZE-L' uses the pre-trained CLIP ViT-L/14,"
                             "if 'SEIZE-g' uses the pre-trained CLIP ViT-g/14,"
                             "if 'SEIZE-G' uses the pre-trained CLIP ViT-G/14,"
                             "if 'SEIZE-CoCa-L' uses the pre-trained CoCa ViT-L/14")
    
    parser.add_argument("--gpt_version", type=str, choices=['gpt-3.5', 'gpt-4', 'gpt-4o'], default='gpt-4o')
    ### 'circo_L_gpt4_2.1_neg_0.13_pos'
    ### 'test_L_momentum_1.5_diff_pos_eql_zero' 'test_L_rank_5e-7' # 'test_L_m_2.1_neg_0.13_pos' 'cirr_G_m_1.0_neg_0.22_pos' 'fashioniq_L_searle_0.0_neg_0.0_pos'
    parser.add_argument("--submission_name", type=str, default='4o_g', help="Filename of the generated submission file") # 'multi_opt_gpt35_15_momentum_0.3' 'multi_opt_gpt35_15_debiased_0.1_sum'
    parser.add_argument("--caption_type", type=str, default='opt', choices=['none', 't5', 'opt'])
    parser.add_argument("--nums_caption", type=int, default=15)
    
    parser.add_argument("--use_momentum_strategy", type=bool, default=True)
    parser.add_argument("--pos_factor", type=float, default=0.13) # 5e-7 0.13 0.22 0.31 0.0005
    parser.add_argument("--neg_factor", type=float, default=2.1) # 5e-7 2.1 1.0 5.0 0.25 
    
    args = parser.parse_args()

    