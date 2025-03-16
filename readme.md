# MyDreamer: PersonalizedText-to-VecorGenerationwithDiffusionModel

# Environment Setup

Clone the repository:
   ```bash
   git clone https://github.com/soyeong-kwon/MyDreamer.git
   sh script/install.sh
   ```

# Pretrained Weights

Download the personalized model weights from **DreamMatcher**. These weights are pretrained DreamBooth weights on the ViCo dataset.  
You can find the pre-trained weights [Link](https://github.com/cvlab-kaist/DreamMatcher).  

After downloading, store the weights in the following directory:
```bash
dreambooth/concept_models/
```

# Running the Model
To run the model, execute:

```bash
sh script/run.sh
```

## Customization Options
You can modify the following parameters to customize your settings:

* num_batch: Batch size for generating graphics.
* prompt_index_from / prompt_index_to: Specify the range of prompt indices you want to generate.
* mode: Set to normal or challenging.
* concept_id: Define which object you want to generate.
* total_step: Total number of steps for optimization.
* flag_prev: Set to true to enable DVSD.
* flag_igs: Set to true to enable DGS.
* gs_type: Define the DGS scheduling scheme.
* bg_lambda: Lambda value for background optimization.
* mprec: Use `no` for float32 precision or `fp16` for float16 precision.

# Evaluation
To evaluate the results, run:

```bash
sh script/eval.sh
```

## Before you run
Please update the following paths in the script:

* `real_folder`: Path to the folder containing real pixel images. (for FID)
* `sample_folder`: Path to the folder containing sampled images. 
* `sample_dm_folder`: Path to the folder containing sampled images. 

Note: The `sample_folder` and `sample_dm_folder` contain the same sampled images but differ in folder structure. You can use `utils/move_samples.py` to create the sample_folder.