
import numpy as np
from tqdm import tqdm
from PIL import Image

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"/mnt/server8_hard3/seokil/samples/main/ImageNet/linear_scheduling/256_tokens/soft_True/add/zero/linear_gate/lr_0.0001/gating_c/DiT-XL-2/0020000-size-256-vae-ema-samples-50000-cfg-1.5-seed-0.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    create_npz_from_sample_folder("/mnt/server8_hard3/seokil/samples/main/ImageNet/linear_scheduling/256_tokens/soft_True/add/zero/linear_gate/lr_0.0001/gating_c/DiT-XL-2/0020000-size-256-vae-ema-samples-50000-cfg-1.5-seed-0/")