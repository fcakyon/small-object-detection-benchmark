_base_ = ["../yolox/yolox_s_8x8_300e_coco.py"]

TAGS = ["yolox", "slice_size=500", "overlap_ratio=025", "75epochs"]
DATA_ROOT = "data/xview/"
BATCH_MULTIPLIER = 4
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
    test_cfg=dict(max_per_img=100),
)

# dataset settings
train_dataset = dict(
    dataset=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/train_500_025.json",
        img_prefix=DATA_ROOT + "sliced/train_images_500_025/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
    ),
)

data = dict(
    samples_per_gpu=8 * BATCH_MULTIPLIER,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_500_025.json",
        img_prefix=DATA_ROOT + "sliced/val_images_500_025/",
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_500_025.json",
        img_prefix=DATA_ROOT + "sliced/val_images_500_025/",
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
                name="yolox_s_500_025",
                tags=TAGS,
            ),
            log_artifact=True,
            out_suffix=(".py"),
        ),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
work_dir = "runs/xview/yolox_s_500_025/"
