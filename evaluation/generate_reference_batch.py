import argparse
from PIL import Image
import numpy as np
import os

# from train import center_crop_arr
from tqdm.contrib.concurrent import process_map


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

IMAGESIZE = None
def resize(img_path):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
    pil_img_crop = center_crop_arr(pil_image=im, image_size=IMAGESIZE)
    return np.array(pil_img_crop)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="file path containing reference dataset")
    parser.add_argument("--img_size", type=int, help="target image size")
    parser.add_argument("--out_path", type=str, help="output path for saving .npz file")

    args = parser.parse_args()
    img_file_paths_list = _list_image_files_recursively(args.data_path)
    global IMAGESIZE
    IMAGESIZE = args.img_size

    result = process_map(resize, img_file_paths_list, max_workers=64)
    np.savez(args.out_path, arr_0=np.stack(result))


if __name__ == "__main__":
    main()
