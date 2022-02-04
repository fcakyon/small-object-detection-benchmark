_base_ = ["../fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py"]

TAGS = ["fcos", "slice_size=300", "overlap_ratio=025", "24epochs"]
DATA_ROOT = "data/xview/"
BATCH_MULTIPLIER = 16
LR_MULTIPLIER = 1
EVAL_INTERVAL = 3
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
    bbox_head=dict(
        num_classes=NUM_CLASSES,
    ),
    # testing settings
    test_cfg=dict(max_per_img=100),
)

# dataset settings
data = dict(
    samples_per_gpu=2 * BATCH_MULTIPLIER,
    workers_per_gpu=2,
    train=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/train_300_025.json",
        img_prefix=DATA_ROOT + "sliced/train_images_300_025/",
    ),
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_300_025.json",
        img_prefix=DATA_ROOT + "sliced/val_images_300_025/",
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_300_025.json",
        img_prefix=DATA_ROOT + "sliced/val_images_300_025/",
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
lr_config = dict(policy="step", warmup="constant", warmup_iters=500, warmup_ratio=1.0 / 3, step=[18])
runner = dict(type="EpochBasedRunner", max_epochs=24)

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
                name="fcos_300_025",
                tags=TAGS,
            ),
            log_artifact=True,
            out_suffix=(".py"),
        ),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth"
work_dir = "runs/xview/fcos_300_025/"
