import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import load_json, save_json
from tqdm import tqdm

# fix the seed
random.seed(13)


def xview_to_coco(
    train_images_dir,
    train_geojson_path,
    output_dir,
    train_split_rate=0.75,
    category_id_remapping=None,
):
    """
    Converts visdrone-det annotations into coco annotation.

    Args:
        train_images_dir: str
            'train_images' folder directory
        train_geojson_path: str
            'xView_train.geojson' file path
        output_dir: str
            Output folder directory
        train_split_rate: bool
            Train split ratio
        category_id_remapping: dict
            Used for selecting desired category ids and mapping them.
            If not provided, xView mapping will be used.
            format: str(id) to str(id)
    """

    # init vars
    category_id_to_name = {}
    with open("xview/xview_class_labels.txt", encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        category_id = line.split(":")[0]
        category_name = line.split(":")[1].replace("\n", "")
        category_id_to_name[category_id] = category_name

    if category_id_remapping is None:
        category_id_remapping = load_json("xview/category_id_mapping.json")
    category_id_remapping

    # init coco object
    coco = Coco()
    # append categories
    for category_id, category_name in category_id_to_name.items():
        if category_id in category_id_remapping.keys():
            remapped_category_id = category_id_remapping[category_id]
            coco.add_category(
                CocoCategory(id=int(remapped_category_id), name=category_name)
            )

    # parse xview data
    coords, chips, classes, image_name_to_annotation_ind = get_labels(
        train_geojson_path
    )
    image_name_list = get_ordered_image_name_list(image_name_to_annotation_ind)

    # convert xView data to COCO format
    for image_name in tqdm(image_name_list, "Converting xView data into COCO format"):
        # create coco image object
        width, height = Image.open(Path(train_images_dir) / image_name).size
        coco_image = CocoImage(file_name=image_name, height=height, width=width)

        annotation_ind_list = image_name_to_annotation_ind[image_name]

        # iterate over image annotations
        for annotation_ind in annotation_ind_list:
            bbox = coords[annotation_ind].tolist()
            category_id = str(int(classes[annotation_ind].item()))
            coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            if category_id in category_id_remapping.keys():
                category_name = category_id_to_name[category_id]
                remapped_category_id = category_id_remapping[category_id]
            else:
                continue
            # create coco annotation and append it to coco image
            coco_annotation = CocoAnnotation(
                bbox=coco_bbox,
                category_id=int(remapped_category_id),
                category_name=category_name,
            )
            if coco_annotation.area > 0:
                coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)

    result = coco.split_coco_as_train_val(train_split_rate=train_split_rate)

    train_json_path = Path(output_dir) / "train.json"
    val_json_path = Path(output_dir) / "val.json"
    save_json(data=result["train_coco"].json, save_path=train_json_path)
    save_json(data=result["val_coco"].json, save_path=val_json_path)


def get_ordered_image_name_list(image_name_to_annotation_ind: Dict):
    image_name_list: List[str] = list(image_name_to_annotation_ind.keys())

    def get_image_ind(image_name: str):
        return int(image_name.split(".")[0])

    image_name_list.sort(key=get_image_ind)

    return image_name_list


def get_labels(fname):
    """
    Gets label data from a geojson label file
    Args:
        fname: file path to an xView geojson label file
    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    Modified from https://github.com/DIUx-xView.
    """
    data = load_json(fname)

    coords = np.zeros((len(data["features"]), 4))
    chips = np.zeros((len(data["features"])), dtype="object")
    classes = np.zeros((len(data["features"])))
    image_name_to_annotation_ind = defaultdict(list)

    for i in tqdm(range(len(data["features"])), "Parsing xView data"):
        if data["features"][i]["properties"]["bounds_imcoords"] != []:
            b_id = data["features"][i]["properties"]["image_id"]
            # https://github.com/DIUx-xView/xView1_baseline/issues/3
            if b_id == "1395.tif":
                continue
            val = np.array(
                [
                    int(num)
                    for num in data["features"][i]["properties"][
                        "bounds_imcoords"
                    ].split(",")
                ]
            )
            chips[i] = b_id
            classes[i] = data["features"][i]["properties"]["type_id"]

            image_name_to_annotation_ind[b_id].append(i)

            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = "None"

    return coords, chips, classes, image_name_to_annotation_ind


if __name__ == "__main__":
    # set 'train_images' folder directory
    train_images_dir = None
    # set train geojson path
    train_geojson_path = None
    # set output folder directory
    output_dir = None

    xview_to_coco(
        train_images_dir=train_images_dir,
        train_geojson_path=train_geojson_path,
        output_dir=output_dir,
    )
