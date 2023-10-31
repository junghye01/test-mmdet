
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=50),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
    visualization=dict(type='DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results

optim_wrapper = dict(  # Optimizer wrapper config
    type='SGD',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.02,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001),  # Weight decay of SGD
    clip_grad=None,  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # Use linear policy to warmup learning rate
        start_factor=0.001, # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=500),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type='MultiStepLR',  # Use multi-step learning rate policy during training
        by_epoch=True,  # The learning rate is updated by epoch
        begin=0,   # Start from the first epoch
        end=12,  # End at the 12th epoch
        milestones=[8, 11],  # Epochs to decay the learning rate
        gamma=0.1)  # The learning rate decay ratio
]