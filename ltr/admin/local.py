class EnvironmentSettings:
    def __init__(self):

        root_path = '/home/sgn/Data1/yan/'

        self.workspace_dir = root_path + 'pytracking-models/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
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
        self.cdtb_dir = root_path + 'Datasets/CDTB/'

        self.lasotdepth_dir = root_path + 'Datasets/EstimatedDepth/LaSOT/'
        self.cocodepth_dir = root_path + 'Datasets/EstimatedDepth/COCO_densedepth/'

        self.depthtrack_dir = root_path + 'Datasets/DeTrack-v1/train_annotated/'
        # self.depthtrack_horizontal_dir = root_path + 'Datasets/DeTrack-v1/train_horizontal_flip/'
        # self.depthtrack_vertical_dir = root_path + 'Datasets/DeTrack-v1/train_vertical_flip/'
