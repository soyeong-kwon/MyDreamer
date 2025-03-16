import os 
import shutil 
import glob 
import cairosvg 
from PIL import Image 
import tqdm 

objects = ["pink_sunglasses", "red_vase", "tortoise_plushy", "wooden_barn", 
           "fat_dog", "brown_dog2", "brown_dog", "brown_cat"]

mode = 'normal'

num_prompts = 8
num_samples = 4 

from_folder = f''
to_folder = f''
to_folder_fid = to_folder.replace('_dm', '_fid')

os.makedirs(to_folder, exist_ok= True)
os.makedirs(to_folder_fid, exist_ok= True)

for object in objects:
    from_object_folder = from_folder + f"/{object}"
    
    for p_i in tqdm.tqdm(range(num_prompts)):
        from_object_p_folder = from_object_folder + f"/{p_i}"
        
        os.makedirs(f'{to_folder}/{object}/output/base/prompt_{p_i}', exist_ok = True)
        os.makedirs(f"{to_folder_fid}/base/{object}", exist_ok= True)
        
        for s_i in range(num_samples):
            input_svg_file = f"{from_object_p_folder}/{s_i}/Final.svg"
            
            from_object_p_s_folder = from_object_p_folder + f"/{s_i}"
            
            output_png_file = f'{to_folder}/{object}/output/base/prompt_{p_i}/{s_i}.png'
            output_png_file_fid = f'{to_folder_fid}/base/{object}_{p_i}_{s_i}.jpg'
            
            cairosvg.svg2png(url=input_svg_file, write_to=output_png_file)
        
            image = Image.open(output_png_file)

            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            background = Image.new('RGB', image.size, (255, 255, 255))
            image_without_transparency = Image.alpha_composite(background.convert('RGBA'), image)

            image_without_transparency = image_without_transparency.convert('RGB')
            image_without_transparency.save(output_png_file)
            
            image_without_transparency.save(output_png_file_fid)