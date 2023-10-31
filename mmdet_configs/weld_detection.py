# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/public/yunvqa/clip/archive/data_completed/'
backend_args = None


train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile', backend_args=backend_args),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=True,  # Whether to use instance mask, True for instance segmentation
        poly2mask=True),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(
        type='Resize',  # Pipeline that resizes the images and their annotations
        scale=(1333, 800),  # The largest scale of the images
        keep_ratio=True  # Whether to keep the ratio between height and width
        ),
    dict(
        type='RandomFlip',  # Augmentation pipeline that flips the images and their annotations
        prob=0.5),  # The probability to flip
    dict(type='PackDetInputs')  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]
test_pipeline = [  # Testing data processing pipeline
    dict(type='LoadImageFromFile', backend_args=backend_args),  # First pipeline to load images from file path
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # Pipeline that resizes the images
    dict(
        type='PackDetInputs',  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(   # Train dataloader config
    batch_size=2,  # Batch size of a single GPU
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(  # training data sampler
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # randomly shuffle the training data in each epoch
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Batch sampler for grouping images with similar aspect ratio into a same batch. It can reduce GPU memory cost.
    dataset=dict(  # Train dataset config
        type=dataset_type,
        data_root=data_root,
        ann_file='annos/train_annotations_full.json',  # Path of annotation file
        data_prefix=dict(img='origin/train/Image/'),  # Prefix of image path
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # Config of filtering images and annotations
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(  # Validation dataloader config
    batch_size=1,  # Batch size of a single GPU. If batch-size > 1, the extra padding area may influence the performance.
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    drop_last=False,  # Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # not shuffle during validation and testing
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annos/valid_annotations_full.json',
        data_prefix=dict(img='origin/val/Image/'),
        test_mode=True,  # Turn on the test mode of the dataset to avoid filtering annotations or images
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader  # Testing dataloader config

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root+'annos/valid_annotations_full.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator  # Testing evaluator config

train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type