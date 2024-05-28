# Evaluation Guide
여기 코드는 dataset 마다 evaluation metric을 재는 것을 목표로 함.
- 지원해야하는 metric들
  - Precision-Recall
  - Inception Score
  - FID

- Table of content
  * [1. FFHQ evaluation.](#1-ffhq-evaluation)
  * [2. ImageNet Evaluation](#2-imagenet-evaluation)


## 0. Installation
여기 폴더에서는 tensorflow를 사용하여 evaluation을 진행하기 때문에, 가급이면 새로운 python environment를 생성하는 것을 추천드립니다.

1. Create new environments

2. Download python packages with this command.
    ```
    python3 -m pip install -r requirements.txt
    ```


## 1. FFHQ evaluation.
- Generate FFHQ reference batch 
  ```
  python generate_reference_batch.py --data_path [FFHQ_directory] --img_size [img_size] --out_path [output path]
  ```
- evaluation 여기서 무조건 GPU 1개로만 해야 함!!
  ```
  python evaluator.py --ref_batch [reference.npz] --sample_batch [generated.npz] --save_result_path [logging path]
  ```


## 2. ImageNet Evaluation
- Download reference batch: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)


## 3. COCO Evaluation
- Download reference stats: [reference stats](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing)
  ```
  python evaluator_t2i.py --ref_stat_path [ref_stat.npz] --sample_path [sample folder]
  ```

## 서버에서 cuda 못찾을때 다음 커멘드로 해결가능
```
for cudnn_so in /usr/lib/python3/dist-packages/tensorflow/libcudnn*; do
  sudo ln -s "$cudnn_so" /usr/lib/x86_64-linux-gnu/
done
```

