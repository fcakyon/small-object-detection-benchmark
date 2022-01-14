_base_ = ["../yolox/yolox_s_8x8_300e_coco.py"]

DATA_ROOT = "data/xview/"
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
        type="YOLOXHead", num_classes=NUM_CLASSES, in_channels=128, feat_channels=128
    ),
)

# dataset settings


train_dataset = dict(
    dataset=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/train_400_0.json",
        img_prefix=DATA_ROOT + "sliced/train_images_400_0/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
    ),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_400_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_400_0/",
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_400_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_400_0/",
    ),
)

# optimizer
# default 8 gpu
optimizer = dict(
    type="SGD",
    lr=0.01 / 8,  # for 1 gpu
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

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
                name="yolox_s_400_0",
                tags=["yolox", "slice_size=400", "overlap_ratio=0"],
            ),
        ),
        dict(
            type="NeptuneLoggerHook",
            init_kwargs=dict(
                project="OBSS-ML/xview",
                name="yolox_s_400_0",
                tags=["yolox", "slice_size=400", "overlap_ratio=0"],
            ),
        ),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
work_dir = "runs/xview/yolox_s_400_0/"
