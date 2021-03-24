from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    data_root = '/home/sgn/Data1/yan/'

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = data_root + 'pytracking-models/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = data_root + 'pytracking-models/result_plots/'
    settings.results_path = data_root + 'pytracking-models/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = ''
    settings.scoremap_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.cdtb_path = data_root + 'Datasets/CDTB/'
    settings.youtubevos_dir = ''
    settings.depthtrack_path = data_root + 'Datasets/DeTrack-v1/test/'

    return settings
