from pytorch_fid import fid_score
import os
import pdb

imagenet_dir = "/mnt/server3_hard2/hongsin/dataset/ImageNet/train"
class_folders = [os.path.join(imagenet_dir, f) for f in os.listdir(imagenet_dir) if os.path.isdir(os.path.join(imagenet_dir, f))]

sfid = 0
count = 0
for class_folder in class_folders:
    sfid += fid_score.main(_path2="/mnt/server8_hard3/seokil/samples/main/ImageNet/linear_scheduling/256_tokens/soft_True/add/zero/linear_gate/lr_0.0001/gating_c/DiT-XL-2/0020000-size-256-vae-ema-samples-50000-cfg-1.5-seed-0/", _path1=class_folder)
    count += 1
print("sFID: ", sfid/count)