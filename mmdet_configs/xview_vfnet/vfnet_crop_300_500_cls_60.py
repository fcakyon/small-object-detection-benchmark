_base_ = ["../vfnet/vfnet_r50_fpn_1x_coco.py"]


EXP_NAME = "vfnet_crop_300_500_cls_60"
DATA_ROOT = "data/xview/"
BATCH_MULTIPLIER = 8
LR_MULTIPLIER = 1
EVAL_INTERVAL = 3
NUM_CLASSES = 60
DATASET_REPEAT = 50
TAGS = ["vfnet", "crop=300_500", "24epochs", f"num_cls={NUM_CLASSES}", f"repeat={DATASET_REPEAT}"]
CLASSES = (
    "Fixed-wing Aircraft",
    "Small Aircraft",
    "Cargo Plane",
    "Helicopter",
    "Passenger Vehicle",
    "Small Car",
    "Bus",
    "Pickup Truck",
    "Utility Truck",
    "Truck",
    "Cargo Truck",
    "Truck w/Box",
    "Truck Tractor",
    "Trailer",
    "Truck w/Flatbed",
    "Truck w/Liquid",
    "Crane Truck",
    "Railway Vehicle",
    "Passenger Car",
    "Cargo Car",
    "Flat Car",
    "Tank car",
    "Locomotive",
    "Maritime Vessel",
    "Motorboat",
    "Sailboat",
    "Tugboat",
    "Barge",
    "Fishing Vessel",
    "Ferry",
    "Yacht",
    "Container Ship",
    "Oil Tanker",
    "Engineering Vehicle",
    "Tower crane",
    "Container Crane",
    "Reach Stacker",
    "Straddle Carrier",
    "Mobile Crane",
    "Dump Truck",
    "Haul Truck",
    "Scraper/Tractor",
    "Front loader/Bulldozer",
    "Excavator",
    "Cement Mixer",
    "Ground Grader",
    "Hut/Tent",
    "Shed",
    "Building",
    "Aircraft Hangar",
    "Damaged Building",
    "Facility",
    "Construction Site",
    "Vehicle Lot",
    "Helipad",
    "Storage Tank",
    "Shipping container lot",
    "Shipping Container",
    "Pylon",
    "Tower",
)

# model settings
model = dict(
    bbox_head=dict(
        num_classes=NUM_CLASSES,
    ),
)

# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(300, 500), allow_negative_crop=True),
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
            ],
            [
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(300, 500), allow_negative_crop=True),
                dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
            ],
            [
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(300, 500), allow_negative_crop=True),
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
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2 * BATCH_MULTIPLIER,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=DATASET_REPEAT,
        dataset=dict(
            type="CocoDataset",
            classes=CLASSES,
            ann_file=DATA_ROOT + "coco/train.json",
            img_prefix=DATA_ROOT + "train_images/",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_400_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_400_0/",
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_400_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_400_0/",
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
lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.1, step=[16, 22])
runner = dict(type="EpochBasedRunner", max_epochs=24)

# logger settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook", reset_flag=False),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth"
work_dir = f"runs/xview/{EXP_NAME}/"
