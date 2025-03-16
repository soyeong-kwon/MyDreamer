import os
import argparse
import torch
from code.personalized import PersonalizedBase
from code.clip_eval import CLIPEvaluator
from code.dino_eval import DINOEvaluator
from statistics import mean
from accelerate.utils import set_seed

set_seed(42)

def main(opt):
    object_name = opt.src_dir.split("/")[-1]
    live_objects = ["black_cat", "brown_cat", "brown_dog2", "brown_dog", "fat_dog"]
    if opt.mode == "challenging":
        if object_name in live_objects:
            with open("/home/s20235025/cvpr2025/dreambooth/inputs/prompts_live_objects_challenging.txt") as file:
                prompts = [line.rstrip() for line in file]
        else:
            with open("/home/s20235025/cvpr2025/dreambooth/inputs/prompts_nonlive_objects_challenging.txt") as file:
                prompts = [line.rstrip() for line in file]
    else:
        if object_name in live_objects:
            with open("/home/s20235025/cvpr2025/dreambooth/inputs/prompts_live_objects.txt") as file:
                prompts = [line.rstrip() for line in file]
        else:
            with open("/home/s20235025/cvpr2025/dreambooth/inputs/prompts_nonlive_objects.txt") as file:
                prompts = [line.rstrip() for line in file]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    src_data_loader = PersonalizedBase(opt.src_dir, size=256, flip_p=0.0, set="eval")
    img_sims_clip = []
    img_sims_dino = []
    txt_sims = []
    clip_evaluator = CLIPEvaluator(device)
    dino_evaluator = DINOEvaluator(device)

    for i, prompt in enumerate(prompts[:opt.num_prompts]):

        prompt = prompt.format(object_name)

        print(f"evaluating prompt {i}: {prompt}")

        trg_dir = os.path.join(opt.trg_dir, f"prompt_{i}")
        if os.path.exists(os.path.join(trg_dir, "samples")):
            trg_dir = os.path.join(trg_dir, "samples")
        trg_data_loader = PersonalizedBase(trg_dir, size=256, flip_p=0.0)

        src_images = [
            torch.from_numpy(src_data_loader[i]["image"]).permute(2, 0, 1)
            for i in range(src_data_loader.num_images)
        ]
        trg_images = [
            torch.from_numpy(trg_data_loader[i]["image"]).permute(2, 0, 1)
            for i in range(trg_data_loader.num_images)
        ]
        src_images = torch.stack(src_images, axis=0)
        trg_images = torch.stack(trg_images, axis=0)

        sim_img_dino = dino_evaluator.img_to_img_similarity(src_images, trg_images)
        sim_img_clip = clip_evaluator.img_to_img_similarity(src_images, trg_images)
        sim_text = clip_evaluator.txt_to_img_similarity(prompt, trg_images)

        sim_img_clip = float(sim_img_clip.cpu().numpy())
        sim_img_dino = float(sim_img_dino.cpu().numpy())
        sim_text = float(sim_text.cpu().numpy())

        img_sims_clip.append(sim_img_clip)
        img_sims_dino.append(sim_img_dino)
        txt_sims.append(sim_text)

    with open(f"{opt.trg_dir}/{opt.concept}_metrics.txt", "w") as f:
        for prompt, img_sim_dino, img_sim_clip, txt_sim in zip(
            prompts, img_sims_dino, img_sims_clip, txt_sims
        ):
            f.write(f"{img_sim_dino} {img_sim_clip} {txt_sim} {prompt}\n")
        avg_img_dino = mean(img_sims_dino)
        f.write(str(avg_img_dino) + "\n")
        avg_img_clip = mean(img_sims_clip)
        f.write(str(avg_img_clip) + "\n")
        avg_txt_clip = mean(txt_sims)
        f.write(str(avg_txt_clip) + "\n")

    with open(f"{opt.trg_dir}/metrics.txt", "w") as f:
        f.write(f"{opt.concept}\n{avg_img_dino}\n{avg_img_clip}\n{avg_txt_clip}")

    print("total dino img sim: ", avg_img_dino)
    print("total clip img sim: ", avg_img_clip)
    print("total txt sim: ", avg_txt_clip)
    print(f"metrics were saved in {opt.trg_dir}/metrics.txt")
    
    return avg_img_dino, avg_img_clip, avg_txt_clip
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_dir",
        type=str,
        help="dir to result images",
    )

    parser.add_argument(
        "--model",
        type=str,
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="mode to evaluate",
        choices=["challenging", "normal"],
    )

    parser.add_argument(
        "--num_prompts",
        type=int,
    )


    args = parser.parse_args()

    vico_concepts = {
        "cat_statue": "cat-toy",
        "elephant_statue": "elephant_statue",
        "duck_toy": "duck_toy",
        "monster_toy": "monster_toy",
        #####################
        "brown_teddybear": "teddybear",
        "tortoise_plushy": "tortoise_plushy",
        "brown_dog": "brown_dog",
        "fat_dog": "fat_dog",
        #####################
        "brown_dog2": "brown_dog2",
        "black_cat": "black_cat",
        "brown_cat": "brown_cat",
        "alarm_clock": "clock",
        #####################
        "pink_sunglasses": "pink_sunglasses",
        "red_teapot": "red_teapot",
        "red_vase": "vase",
        "wooden_barn": "barn",
    }

    total_img_dino = 0 
    total_img_clip = 0 
    total_txt_clip = 0
    
    total_concet_dir = os.listdir(args.result_dir)
    total_concet_dir = [item for item in total_concet_dir if item in vico_concepts]
    # total_concet_dir = ["pink_sunglasses", "red_vase", "tortoise_plushy", "wooden_barn"]
    total_concet_dir_num = len(total_concet_dir)
    
    print(total_concet_dir, total_concet_dir_num)

    for concept_dir in total_concet_dir:
        if not os.path.isdir(os.path.join(args.result_dir, concept_dir)):
            continue

        print(f"processing {concept_dir}")
        concept = concept_dir.split("-")[0]

        for target in [args.model]:
            args.concept = concept
            args.src_dir = f"/home/s20235025/cvpr2025/dreambooth/inputs/{vico_concepts[concept_dir]}"
            args.trg_dir = f"{args.result_dir}/{concept_dir}/output/{target}"
            
            avg_img_dino, avg_img_clip, avg_txt_clip = main(args)
            
            total_img_dino = total_img_dino + avg_img_dino 
            total_img_clip = total_img_clip + avg_img_clip
            total_txt_clip = total_txt_clip + avg_txt_clip

    with open(f"{args.result_dir}/{target}_metrics.txt", "w") as f:
        f.write(f"final for {total_concet_dir_num} concepts\n")
        f.write(f'final_img_dino: {total_img_dino / total_concet_dir_num}\n')
        f.write(f'final_img_clip: {total_img_clip / total_concet_dir_num}\n')
        f.write(f'final_txt_clip: {total_txt_clip / total_concet_dir_num}\n')
        
    print(f'final_img_dino: {total_img_dino / total_concet_dir_num}')
    print(f'final_img_clip: {total_img_clip / total_concet_dir_num}')
    print(f'final_txt_clip: {total_txt_clip / total_concet_dir_num}')
    print(f"metrics were saved in {args.result_dir}/metrics.txt")
    
#   python evaluate.py --result_dir "samples_dm/" --mode "normal"