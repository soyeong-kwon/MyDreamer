import torch.multiprocessing as mp
import omegaconf
import os
import sys
from datetime import datetime
import json
import torch

from accelerate.utils import set_seed
import hydra

import torch 
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from mydreamer.utils import render_batch_wrap, get_seed_range
from mydreamer.pipelines.MyDreamer_pipeline import MyDreamerPipeline
from mydreamer.painter import DiffusionPipeline
import argparse
from functools import partial

def set_prompt(object, mode):
    txt_path = './dreambooth/inputs'
        
    vico_dict = {'cat_statue': ('cat_statue', 'cat-toy', False),
        'elephant_statue': ('elephant_statue', 'elephant_statue', False),
        'duck_toy': ('duck_toy', 'duck_toy', False),
        'monster_toy': ('monster_toy', 'monster_toy', False),
        'brown_teddybear': ('brown_teddybear', 'teddybear', False),
        'tortoise_plushy': ('tortoise_plushy', 'tortoise_plushy', False),
        'brown_dog': ('brown_dog', 'brown_dog', True),
        'fat_dog': ('fat_dog', 'fat_dog', True),
        'brown_dog2': ('brown_dog2', 'brown_dog2', True),
        'black_cat': ('black_cat', 'black_cat', True),
        'brown_cat': ('brown_cat', 'brown_cat', True),
        'alarm_clock': ('alarm_clock', 'clock', False),
        'pink_sunglasses': ('pink_sunglasses', 'pink_sunglasses', False),
        'red_teapot': ('red_teapot', 'red_teapot', False),
        'red_vase': ('red_vase', 'vase', False),
        'wooden_barn': ('wooden_barn', 'barn', False)}

    item = vico_dict.get(object)
    concept, concept_orig_name, is_live = item

    if mode == "challenging":
        if is_live:
            with open(f"{txt_path}/prompts_live_objects_challenging.txt", "r") as fin:
                prompts = fin.read()
                prompts = prompts.split("\n")
        else:
            with open(f"{txt_path}/prompts_nonlive_objects_challenging.txt", "r") as fin:
                prompts = fin.read()
                prompts = prompts.split("\n")
    else:
        if is_live:
            with open(f"{txt_path}/prompts_live_objects.txt", "r") as fin:
                prompts = fin.read()
                prompts = prompts.split("\n")
        else:
            with open(f"{txt_path}/prompts_nonlive_objects.txt", "r") as fin:
                prompts = fin.read()
                prompts = prompts.split("\n")
    
    return prompts        

def sample(args, concept, prompts):
    
    result_d_path = args.result_dir + f"/flag_{args.flag_prev}_{args.gs_type}_{args.mode}/{concept}"
    
    concept_model_path = './dreambooth/concept_models' 
    concept_dir = f"{concept_model_path}/{concept}/checkpoints/diffusers"
    print(concept_dir)
    
    ref_prompt = f"a photo of sks {concept}"
    
    for p_i, prompt in enumerate(prompts):
        p_i = p_i + args.prompt_index_from
        result_path = f"{result_d_path}/{p_i}"
        gen_prompt = prompt.format(f"sks {concept}")
        ref_prompt = ref_prompt.format(f"sks {concept}")
        
        cur_token_idx = (gen_prompt.split(" ").index(f"{concept}") + 1)  # 1-indexed because of <sos>

        cfg = omegaconf.OmegaConf.load("./conf/config.yaml")
        set_seed(cfg.seed)

        cfg.output_dir = result_path
        cfg.prompt = gen_prompt
        cfg.token_ind = cur_token_idx

        ## argument 순서대로 parameter 설정 ## 
        cfg.num_paths = args.total_path 
        cfg.run_type.gs_range = args.gs_range
        cfg.state.mprec = args.mprec 
        
        cfg.x = omegaconf.OmegaConf.load("./conf/x/mydreamer.yaml")
        cfg.x.sive_model_cfg.model_id = concept_dir
        cfg.x.vpsd_model_cfg.model_id = concept_dir
        cfg.x.sive.bg.num_paths = args.total_path // 2
        cfg.x.sive.fg.num_paths = args.total_path // 2
        cfg.x.vpsd.num_iter = args.total_step
        
        # quick 
        # cfg.x.sive_model_cfg.num_inference_steps = 5
        
        # save config 
        os.makedirs(result_path + "/conf", exist_ok= True)
        os.makedirs(result_path + "/conf/x", exist_ok= True)

        omegaconf.OmegaConf.save(cfg, result_path + "/conf/config.yaml")
        omegaconf.OmegaConf.save(cfg.x, result_path + "/conf/x/mydreamer.yaml")
        print("Configuration saved successfully.") 

        pipe = MyDreamerPipeline(cfg)
        pipe.flag_prev = args.flag_prev 
        pipe.flag_igs = args.flag_igs
        pipe.gs_type = args.gs_type
        pipe.bg_lambda = args.bg_lambda
        
        for batch_idx in range(args.num_batch):
            os.makedirs(result_path + f"/{batch_idx}", exist_ok= True)
            
            print("==> FIVE stage")
            merged_svg_path, attn_path, gt_path = pipe.FIVE_stage(text_prompt= cfg.prompt, batch_idx = batch_idx)
            
            print('==> Optimize Vector')
            pipe.SD_stage(text_prompt= cfg.prompt, 
                          prev_prompt= ref_prompt, 
                          batch_idx = batch_idx, 
                          merged_svg_path = merged_svg_path,
                          attn_path = attn_path,
                          gt_path = gt_path)

        del pipe 
        torch.cuda.empty_cache()

def main():
    now = datetime.now()
    formatted_now = now.strftime("%y%m%d_%H%M")

    parser = argparse.ArgumentParser(description="Process some command line arguments.")
    parser.add_argument('--num_batch', type = int, default= 4)
    parser.add_argument('--prompt_index_from', type = int, default= 0)
    parser.add_argument('--prompt_index_to', type = int, default = 8)
    parser.add_argument('--mode', type = str)
    parser.add_argument('--concept_id', type = int)
    parser.add_argument('--result_dir', type=str, help="result_dir")
    
    parser.add_argument('--total_path', type = int, default= 512)
    parser.add_argument('--total_step', type = int, default= 1000)
    parser.add_argument('--gs_range', type = str, default='7.5_100')
    parser.add_argument('--mprec', type = str, default='fp16') ## 
    
    parser.add_argument('--flag_prev', action='store_true', help="Set this flag to True")
    parser.add_argument('--flag_igs', action='store_true')
    parser.add_argument('--gs_type', type = str, default= 'fixed')
    parser.add_argument('--bg_lambda', type = float)

    args = parser.parse_args()
    args.result_dir = args.result_dir + f"/{formatted_now}"
    vico_concepts = ['red_vase', 'pink_sunglasses', 'tortoise_plushy', 'wooden_barn',
                     'fat_dog', 'brown_dog2', 'brown_dog', 'brown_cat']

    concept = vico_concepts[args.concept_id]
    prompts = set_prompt(concept, mode = args.mode)
    prompts = prompts[args.prompt_index_from:args.prompt_index_to]
    print(prompts)
    sample(args, concept, prompts)

if __name__ == '__main__':
    
    main()
