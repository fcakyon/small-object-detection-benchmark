_base_ = ["../detr/detr_r50_8x2_150e_coco.py"]

TAGS = ["detr", "slice_size=300", "overlap_ratio=025", "150epochs"]
DATA_ROOT = "data/xview/"
BATCH_MULTIPLIER = 8
LR_MULTIPLIER = 1
EVAL_INTERVAL = 10
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

model = dict(
    bbox_head=dict(
        type="DETRHead",
        num_query=100,
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
    type="AdamW",
    lr=0.0001 / 8 * BATCH_MULTIPLIER * LR_MULTIPLIER,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}),
)

checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_optimizer=False)
evaluation = dict(interval=EVAL_INTERVAL, metric="bbox", save_best="auto")

# learning policy
lr_config = dict(policy="step", step=[100])
runner = dict(type="EpochBasedRunner", max_epochs=150)

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
                name="detr_r50_300_025",
                tags=TAGS,
            ),
            log_artifact=True,
            out_suffix=(".py"),
        ),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"
work_dir = "runs/xview/detr_r50_300_025/"
