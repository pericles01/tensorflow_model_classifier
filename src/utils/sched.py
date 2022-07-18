"""Description
   -----------
This script includes all functions that are related to training scheduling
and learning rate decay methods
"""

import math
import tensorflow as tf
import numpy as np


def step_lr_decay(args):
    """
    Step learning rate decay
    :param args
    :return: updated learning rate
    """
    if args.lr_decay:
        def lr_scheduler(epoch, lr):
            """step decay with a rate of >drop< every >drop_interval (currently fixed)<"""
            initial_lr = args.lr
            drop = 0.5
            drop_interval = args.epochs // 10
            return initial_lr * math.pow(drop, math.floor((1 + epoch) / drop_interval))
    else:
        def lr_scheduler(epoch, lr):
            return lr
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


def cosine_decay_with_warmup(global_step, warmup_steps, total_steps, lrs):
    """
    Cosine learning rate decay with warmup stage
    :param global_step: current number of training step (batch-wise)
    :param warmup_steps: number of warmup steps (batch-wise)
    :param total_steps:  total number of training steps (batch-wise)
    :param lrs: list of minimum and maximum learning rates
    :return: updated learning rate
    """
    lr_init = lrs[0]
    lr_end = lrs[1]
    if global_step < warmup_steps:
        lr = global_step / warmup_steps * lr_init
    else:
        lr = lr_end + 0.5 * (lr_init - lr_end) * (
            (1 + tf.cos((global_step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )
    return lr