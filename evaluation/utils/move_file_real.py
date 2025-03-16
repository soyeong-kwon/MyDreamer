import os 
import shutil 
import glob 
import cairosvg 
from PIL import Image 

objects = ["pink_sunglasses", "red_vase", "tortoise_plushy", "wooden_barn", 
           "fat_dog", "brown_dog2", "brown_dog", "brown_cat"]
objects = objects[:-2]

mode = 'normal'

num_prompts = 8
num_samples = 4 

from_folder = f'/home/s20235025/cvpr2025/personalization/evaluation/samples/{mode}_real_6obj'
to_folder = f'/home/s20235025/cvpr2025/personalization/evaluation/samples/{mode}_real_6obj'

os.makedirs(to_folder, exist_ok= True)

for object in objects:
    from_object_folder = from_folder + f"/{object}_{mode}"
    
    for p_i in range(num_prompts):
        from_object_p_folder = from_object_folder + f"/{p_i}"
        
        for s_i in range(num_samples):
            sampled_real_file = f"{from_object_p_folder}/sd42-sive-iconography-P512-RePath/select_sample_{s_i}.png"
            output_png_file_fid = f'{to_folder}/{object}_{p_i}_{s_i}.jpg'

            shutil.copyfile(sampled_real_file, output_png_file_fid)
