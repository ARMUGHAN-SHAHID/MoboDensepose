from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import pprint
import sys

# from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
# from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
# import detectron.utils.c2 as c2_utils
# import detectron.utils.train

import detectron.roi_data.minibatch as roi_data_minibatch
from detectron.core.config import get_output_dir
from detectron.datasets.roidb import combined_roidb_for_training


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    # Initialize C2
    # workspace.GlobalInit(
    #     ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1']
    # )
    # Set up logging and load config options
    logger = setup_logging(__name__)
    logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
    print ("1.   ==============>")
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
        print ("2.   ==============>")
    if args.opts is not None:
        print ("3.   ==============>")
        merge_cfg_from_list(args.opts)
    print ("4.   ==============>")
    assert_and_infer_cfg()
    print ("5.   ==============>")
    # logger.info('Training with config:')
    # logger.info(pprint.pformat(cfg))
    # Note that while we set the numpy random seed network training will not be
    # deterministic in general. There are sources of non-determinism that cannot
    # be removed with a reasonble execution-speed tradeoff (such as certain
    # non-deterministic cudnn functions).
    np.random.seed(cfg.RNG_SEED)

    # Execute the training run
    # checkpoints = detectron.utils.train.train_model()
    # Test the trained model
    # if not args.skip_test:
    #     test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)

    print ("6.   ==============>")
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    print ("7.   ==============>")
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )
    print (type(roidb))
    print (len(roidb))
    for i in range(20):
    	print (roidb[i]['boxes'].shape)
    print ("8.   ==============>")
    # blob_names = roi_data_minibatch.get_minibatch_blob_names(is_training=True)
    # print (blob_names)


if __name__ == '__main__':
    main()