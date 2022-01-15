_base_ = ["../fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py"]

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
        num_classes=NUM_CLASSES,
    )
)

# dataset settings
data = dict(
    samples_per_gpu=2 * 8,
    workers_per_gpu=2,
    train=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/train_500_0.json",
        img_prefix=DATA_ROOT + "sliced/train_images_500_0/",
    ),
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_500_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_500_0/",
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "sliced/val_500_0.json",
        img_prefix=DATA_ROOT + "sliced/val_images_500_0/",
    ),
)

# optimizer
# default 8 gpu
# /8 for 1 gpu
optimizer = dict(lr=0.01 / 8 * 8, paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))

evaluation = dict(metric="bbox", save_best="auto")

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
                name="fcos_500_0",
                tags=["fcos", "slice_size=500", "overlap_ratio=0"],
            ),
        ),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth"
work_dir = "runs/xview/fcos_500_0/"
