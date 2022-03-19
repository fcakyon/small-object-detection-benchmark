_base_ = ["../fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py"]

TAGS = ["fcos", "crop=480_960", "24epochs", "num_cls=60", "repeat=5"]
EXP_NAME = "fcos_crop_480_960_cls_60"
DATA_ROOT = "data/visdrone2019/"
BATCH_MULTIPLIER = 16
LR_MULTIPLIER = 1
EVAL_INTERVAL = 3
NUM_CLASSES = 10
CLASSES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")

# model settings
model = dict(
    bbox_head=dict(
        num_classes=NUM_CLASSES,
    ),
)

# dataset settings
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(480, 960), allow_negative_crop=True),
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
            ],
            [
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(480, 960), allow_negative_crop=True),
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
            ],
            [
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
            ],
        ],
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2 * BATCH_MULTIPLIER,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type="CocoDataset",
            classes=CLASSES,
            ann_file=DATA_ROOT + "coco/train.json",
            img_prefix=DATA_ROOT + "VisDrone2019-DET-train/",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_640_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_640_0/",
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_640_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_640_0/",
        pipeline=test_pipeline,
    ),
)

# optimizer
# default 8 gpu
# /8 for 1 gpu
optimizer = dict(
    lr=0.01 / 8 * BATCH_MULTIPLIER * LR_MULTIPLIER, paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0)
)

checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_optimizer=False)
evaluation = dict(interval=EVAL_INTERVAL, metric="bbox", save_best="auto")

# learning policy
lr_config = dict(policy="step", warmup="constant", warmup_iters=500, warmup_ratio=1.0 / 3, step=[16, 22])
runner = dict(type="EpochBasedRunner", max_epochs=24)

# logger settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook", reset_flag=False),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth"
work_dir = f"runs/visdrone/{EXP_NAME}/"
