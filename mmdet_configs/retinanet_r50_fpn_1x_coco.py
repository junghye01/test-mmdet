_base_ = [
    './retinanet_r50_fpn.py',
    './weld_detection.py',
    './schedule_2x.py', './default_runtime_retina.py'
]

# optimizer

#optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
