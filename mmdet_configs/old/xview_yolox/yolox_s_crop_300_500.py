_base_ = "../yolox/yolox_s_8x8_300e_coco.py"

TAGS = ["yolox", "crop=300_500", "75epochs"]
EXP_NAME = "yolox_s_crop_300_500"
DATA_ROOT = "D:/Data/ObjectDetection/xview/"
BATCH_MULTIPLIER = 4
MAX_DET_PER_IMAGE = 100
EVAL_INTERVAL = 5
NUM_CLASSES = 60
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
    bbox_head=dict(type="YOLOXHead", num_classes=NUM_CLASSES, in_channels=128, feat_channels=128),
    # testing settings
    test_cfg=dict(max_per_img=MAX_DET_PER_IMAGE),
)

# dataset settings
img_scale = (640, 640)
train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(type="RandomAffine", scaling_ratio_range=(0.1, 2), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="CocoDataset",
        ann_file=DATA_ROOT + "coco/train.json",
        img_prefix=DATA_ROOT + "train_images/",
        pipeline=[
            dict(type="LoadImageFromFile", image_backend="pillow"),
            dict(type="LoadAnnotations", with_bbox=True),
            # dict(type="RandomCrop", crop_type="absolute_range", crop_size=(300, 500), allow_negative_crop=True),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=8 * BATCH_MULTIPLIER,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
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
    type="SGD",
    lr=0.01 / 8 * BATCH_MULTIPLIER,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

max_epochs = 75
num_last_epochs = max_epochs

# learning policy
lr_config = dict(
    _delete_=True,
    policy="YOLOX",
    warmup="exp",
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,  # 1 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05,
)

runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="SyncNormHook", num_last_epochs=num_last_epochs, interval=EVAL_INTERVAL, priority=48),
    dict(type="ExpMomentumEMAHook", resume_from=None, momentum=0.0001, priority=49),
]

checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_optimizer=False)
evaluation = dict(interval=EVAL_INTERVAL, metric="bbox", save_best="auto")

# logger settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook", reset_flag=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="xview",
                entity="fca",
                name=EXP_NAME,
                tags=TAGS,
            ),
            log_artifact=True,
            out_suffix=(".py"),
        ),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
work_dir = f"runs/xview/{EXP_NAME}/"
