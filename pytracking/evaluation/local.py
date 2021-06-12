from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.cdtb_path = '/home/sgn/Data1/yan/Datasets/CDTB/'
    settings.cdtb_st_path = '/home/sgn/Data1/yan/Datasets/CDTB-ST'
    settings.davis_dir = ''
    settings.depthtrack_path = '/home/sgn/Data1/yan/Datasets/DepthTrack-v1/test/'
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = ''    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/yan/Data2/DeT/pytracking/result_plots/'
    settings.results_path = '/home/sgn/Data1/yan/DeT/tracking_results/CDTB-ST/'# '/home/yan/Data2/DeT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/yan/Data2/DeT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
