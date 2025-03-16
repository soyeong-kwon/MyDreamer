from pytorch_image_generation_metrics import get_inception_score_from_directory
import torch
from accelerate.utils.random import set_seed
import argparse

parser = argparse.ArgumentParser(description="Process some command line arguments.")
parser.add_argument('--folder', type = str)

# 명령줄 인자를 파싱합니다.
args = parser.parse_args()

set_seed(42)

path = args.folder

IS, IS_std = get_inception_score_from_directory(
        path=path, batch_size=1)

print(IS, path)
