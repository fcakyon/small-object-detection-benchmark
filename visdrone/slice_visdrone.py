import fire
from sahi.scripts.slice_coco import slice
from tqdm import tqdm

SLICE_SIZE_LIST = [480, 640]
OVERLAP_RATIO_LIST = [0, 0.25]
IGNORE_NEGATIVE_SAMPLES = False


def slice_visdrone(image_dir: str, dataset_json_path: str, output_dir: str):
    total_run = len(SLICE_SIZE_LIST) * len(OVERLAP_RATIO_LIST)
    current_run = 1
    for slice_size in SLICE_SIZE_LIST:
        for overlap_ratio in OVERLAP_RATIO_LIST:
            tqdm.write(
                f"{current_run} of {total_run}: slicing for slice_size={slice_size}, overlap_ratio={overlap_ratio}"
            )
            slice(
                image_dir=image_dir,
                dataset_json_path=dataset_json_path,
                output_dir=output_dir,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            current_run += 1


if __name__ == "__main__":
    fire.Fire(slice_visdrone)
