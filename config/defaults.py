"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# -----------------------------------------------------------------------------
# BN options
# -----------------------------------------------------------------------------
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.ROOT = "./data"

# Number of data loader workers per training process.
_C.DATA.NUM_WORKERS = 2

# Load data to pinned host memory.
_C.DATA.PIN_MEMORY = True

# Dataset class name
_C.DATA.DATASET = "ExampleDataset"

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name
_C.MODEL.MODEL_NAME = "ResNet18"

# -----------------------------------------------------------------------------
# Solver options
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.01

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 128

# Resume from checkpoint
_C.TRAIN.RESUME = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = './cache/res18.conv2d.165.0220.pth'

# Checkpoint dir.
_C.TRAIN.CHECKPOINT_DIR = './checkpoint'

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
