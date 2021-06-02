from yacs.config import CfgNode as CN

from maskrcnn_benchmark.config import cfg

# default detector config (Rewrite some of them in cfg)
cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN'
cfg.MODEL.BACKBONE.CONV_BODY = 'DLA-34-FPN'

cfg.MODEL.RPN.USE_FPN = True
cfg.MODEL.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
cfg.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 300
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 300

cfg.MODEL.ROI_HEADS.USE_FPN = True
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

cfg.MODEL.BOX_ON = True
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
cfg.MODEL.ROI_BOX_HEAD.PREDICTOR = "FPNPredictor"
cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024

# DLA
cfg.MODEL.DLA = CN()
cfg.MODEL.DLA.DLA_STAGE2_OUT_CHANNELS = 64
cfg.MODEL.DLA.DLA_STAGE3_OUT_CHANNELS = 128
cfg.MODEL.DLA.DLA_STAGE4_OUT_CHANNELS = 256
cfg.MODEL.DLA.DLA_STAGE5_OUT_CHANNELS = 512
cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS = 128
cfg.MODEL.DLA.STAGE_WITH_DCN = (False, False, False, False, False, False)

# TRACK branch
cfg.MODEL.TRACK_ON = True
cfg.MODEL.TRACK_HEAD = CN()
cfg.MODEL.TRACK_HEAD.TRACKTOR = False
cfg.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
cfg.MODEL.TRACK_HEAD.POOLER_RESOLUTION = 15
cfg.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO = 2

cfg.MODEL.TRACK_HEAD.PAD_PIXELS = 512
# the times of width/height of search region comparing to original bounding boxes
cfg.MODEL.TRACK_HEAD.SEARCH_REGION = 2.0
# the minimal width / height of the search region
cfg.MODEL.TRACK_HEAD.MINIMUM_SREACH_REGION = 0
cfg.MODEL.TRACK_HEAD.MODEL = 'EMM'

# solver params
cfg.MODEL.TRACK_HEAD.TRACK_THRESH = 0.4
cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH = 0.6
cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = 0.4
# maximum number of frames that a track can be dormant
cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES = 1

# track proposal sampling
cfg.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE = 256
cfg.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD = 0.65
cfg.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD = 0.35

cfg.MODEL.TRACK_HEAD.IMM = CN()
# the feature dimension of search region (after fc layer)
# in comparison to that of target region (after fc layer)
cfg.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM_MULTIPLIER = 2
cfg.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM = 256

cfg.MODEL.TRACK_HEAD.EMM = CN()
# Use_centerness flag only activates during inference
cfg.MODEL.TRACK_HEAD.EMM.USE_CENTERNESS = True
cfg.MODEL.TRACK_HEAD.EMM.POS_RATIO = 0.25
cfg.MODEL.TRACK_HEAD.EMM.HN_RATIO = 0.25
cfg.MODEL.TRACK_HEAD.EMM.TRACK_LOSS_WEIGHT = 1.
# The ratio of center region to be positive positions
cfg.MODEL.TRACK_HEAD.EMM.CLS_POS_REGION = 0.8
# The lower this weight, it allows large motion offset during inference
# Setting this param to be small (e.g. 0.1) for datasets that have fast motion,
# such as caltech roadside pedestrian
cfg.MODEL.TRACK_HEAD.EMM.COSINE_WINDOW_WEIGHT = 0.4

# all video-related parameters
cfg.VIDEO = CN()
# the length of video clip for training/testing
cfg.VIDEO.TEMPORAL_WINDOW = 8
# the temporal sampling frequency for training
cfg.VIDEO.TEMPORAL_SAMPLING = 4
cfg.VIDEO.RANDOM_FRAMES_PER_CLIP = 2

#Inference
cfg.INFERENCE = CN()
cfg.INFERENCE.USE_GIVEN_DETECTIONS = False
# The length of clip per forward pass
cfg.INFERENCE.CLIP_LEN = 1

#Solver
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = 16

#Input
cfg.INPUT.MOTION_LIMIT = 0.1
cfg.INPUT.COMPRESSION_LIMIT = 50
cfg.INPUT.MOTION_BLUR_PROB = 0.5
cfg.INPUT.AMODAL = False

# Root directory of datasets
cfg.DATASETS.ROOT_DIR = ''
