from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/yans/pytracking-models/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/yan2/Data2/pytracking-models/pytracking/result_plots/'
    settings.results_path = '/home/yan/Data2/pytracking-models/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/yan/Data2/pytracking-models/pytracking/segmentation_results/'
    # settings.scoremap_path = '/home/yan/Data2/pytracking-models/pytracking/scoremap_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = '/home/yan/Data2/vot-workspace/sequences'
    settings.cdtb_path = '/home/yan/Data2/vot-workspace/sequences'
    settings.youtubevos_dir = ''
    settings.depthtrack_path = '/home/yan/Data2/vot-workspace-DepthTrack/sequences/'
    return settings
