class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yan/Data2/DeT-models/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.cdtb_dir = ''
        self.depthtrack_dir = '/home/yan/Data4/Datasets/DepthTrack/train/'
        self.trackingnetdepth_dir = '/home/yan/Data4/Datasets/EstimatedDepth/TrackingNet/'
        self.lasotdepth_dir = '/home/yan/Data4/Datasets/EstimatedDepth/LaSOT/'
        self.cocodepth_dir = '/home/yan/Data4/Datasets/EstimatedDepth/COCO/'
