# config.py
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[10, 20], [32, 64], [128, 256]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'num_classes':2
}

cfg_slim = {
    'name': 'slim',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300,
    'num_classes':2
}

cfg_rfb = {
    'name': 'RFB',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300,
    'num_classes':2
}

cfg_blaze = {
    'name': 'Blaze',
    # origin anchor
    # 'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
    # kmeans and evolving for 640x640
    'min_sizes': [[8, 11], [14, 19, 26, 38, 64, 149]], 
    'steps': [8, 16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1,
    'cls_weight': 6,
    'landm_weight': 0.1, 
    'gpu_train': True,
    'batch_size': 256,
    'ngpu': 1,
    'epoch': 200,
    'decay1': 130,
    'decay2': 160,
    'decay3': 175,
    'decay4': 185,
    'image_size': 320,
    'num_classes':2
}



