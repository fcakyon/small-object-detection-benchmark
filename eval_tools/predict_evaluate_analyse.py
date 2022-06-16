from pathlib import Path

from sahi.predict import predict
from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate;

MODEL_PATH = ""
MODEL_CONFIG_PATH = ""
EVAL_IMAGES_FOLDER_DIR = ""
EVAL_DATASET_JSON_PATH = ""
INFERENCE_SETTING = "XVIEW_SAHI_FI_PO"
EXPORT_VISUAL = False

############ dont change below #############

INFERENCE_SETTING_TO_PARAMS = {
    "XVIEW_SAHI": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0,
    },
    "XVIEW_SAHI_PO": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0.25,
    },
    "XVIEW_SAHI_FI": {
        "no_standard_prediction": False,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0,
    },
    "XVIEW_SAHI_FI_PO": {
        "no_standard_prediction": False,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0.25,
    },
    "VISDRONE_FI": {
        "no_standard_prediction": False,
        "no_sliced_prediction": True,
        "slice_size": 640,
        "overlap_ratio": 0,
    },
    "VISDRONE_SAHI": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 640,
        "overlap_ratio": 0,
    },
    "VISDRONE_SAHI_PO": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 640,
        "overlap_ratio": 0.25,
    },
    "VISDRONE_SAHI_FI": {
        "no_standard_prediction": False,
        "no_sliced_prediction": False,
        "slice_size": 640,
        "overlap_ratio": 0,
    },
    "VISDRONE_SAHI_FI_PO": {
        "no_standard_prediction": False,
        "no_sliced_prediction": False,
        "slice_size": 640,
        "overlap_ratio": 0.25,
    },
}

setting_params = INFERENCE_SETTING_TO_PARAMS[INFERENCE_SETTING]

result = predict(
    model_type="mmdet",
    model_path=MODEL_PATH,
    model_config_path=MODEL_CONFIG_PATH,
    model_confidence_threshold=0.01,
    model_device="cuda:0",
    model_category_mapping=None,
    model_category_remapping=None,
    source=EVAL_IMAGES_FOLDER_DIR,
    no_standard_prediction=setting_params["no_standard_prediction"],
    no_sliced_prediction=setting_params["no_sliced_prediction"],
    image_size=None,
    slice_height=setting_params["slice_size"],
    slice_width=setting_params["slice_size"],
    overlap_height_ratio=setting_params["overlap_ratio"],
    overlap_width_ratio=setting_params["overlap_ratio"],
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False,
    novisual=not EXPORT_VISUAL,
    dataset_json_path=EVAL_DATASET_JSON_PATH,
    project="runs/predict_eval_analyse",
    name=INFERENCE_SETTING,
    visual_bbox_thickness=None,
    visual_text_size=None,
    visual_text_thickness=None,
    visual_export_format="png",
    verbose=1,
    return_dict=True,
    force_postprocess_type=True,
)

result_json_path = str(Path(result["export_dir"]) / "result.json")

evaluate(
    dataset_json_path=EVAL_DATASET_JSON_PATH,
    result_json_path=result_json_path,
    classwise=True,
    max_detections=500,
    return_dict=False,
)

analyse(
    dataset_json_path=EVAL_DATASET_JSON_PATH,
    result_json_path=result_json_path,
    max_detections=500,
    return_dict=False,
)
