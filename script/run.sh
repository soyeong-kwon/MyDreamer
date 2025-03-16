gs_type="cosine" # or linear
lambda=0.5

conda activate mydreamer

python main.py \
    --result_dir sample \
    --num_batch 4 \
    --prompt_index_from 8 \
    --prompt_index_to 16 \
    --mode normal \
    --concept_id 7 \
    --flag_prev \
    --flag_igs \
    --gs_type $gs_type \
    --bg_lambda $lambda
