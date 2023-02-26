import numpy as np
from nuscenes.nuscenes import NuScenes

# 加载数据集
nusc = NuScenes(version='v1.0-mini',
                dataroot='/home/wn/datasets/nuscenes',
                verbose=True)

# # get data by scene
# scenes = nusc.scene

# samples = []
# annos = []

# for scene in scenes:
#     sample_token = scene['first_sample_token']
#     while sample_token:
#         sample = nusc.get('sample', sample_token)
#         anno_tokens = sample['anns']
#         #     anno = nusc.get('sample_annotation', anno_token[0])
#         #     print(anno.keys())
#         #     break
#         # break
#         annos.extend(anno_tokens)
#         samples.append(sample_token)
#         sample_token = sample['next']

# print(f'num_samples: {len(samples)}')
# print(f'num_annos: {len(annos)}')
# print(f'num_scenes: {len(scenes)}')

# # get data by sample
# samples = nusc.sample

# annos = []
# scenes = set()

# for sample in samples:
#     anno_tokens = sample['anns']
#     scene_token = sample['scene_token']

#     annos.extend(anno_tokens)
#     scenes.add(scene_token)

# print(f'num_samples: {len(samples)}')
# print(f'num_annos: {len(annos)}')
# print(f'num_scenes: {len(scenes)}')

# # 获取所有样本
samples = [samp for samp in nusc.sample]
first_sample = samples[0]

# # 按场景和时间排序
samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

# 获取初始ego_pose
ego_pose = nusc.get(
    'ego_pose',
    nusc.get('sample_data',
             first_sample['data']['LIDAR_TOP'])['ego_pose_token'])

# ego_pose = nusc.get(
#     'ego_pose',
#     nusc.get('sample_data', samples[0]['data']['LIDAR_TOP'])['ego_pose_token'])

# sample_data_record = nusc.get('sample_data', samples[0]['data']['CAM_FRONT'])
# ego_motion = nusc.get(
#     'ego_motion',
#     nusc.get('sample_data',
#              samples[0]['data']['LIDAR_TOP'])['ego_motion_token'])

# initial_state = np.array([
#     ego_pose['translation'][0], ego_pose['translation'][1],
#     np.arctan2(ego_pose['rotation'][1, 0], ego_pose['rotation'][0, 0]),
#     ego_motion['translation'][0], ego_motion['translation'][1],
#     np.arctan2(ego_motion['rotation'][1, 0], ego_motion['rotation'][0, 0])
# ])

# # 初始化卡尔曼滤波器
# dt = 0.1  # 采样时间间隔
# F = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
#               [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
# H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
# P = np.eye(6) * 100  # 状态估计协方差矩阵
# Q = np.eye(6) * 0.1  # 系统噪声协方差矩阵
# R = np.eye(3) * 10  # 观测噪声协方差矩阵
# x_hat = initial_state.reshape(6, 1)  # 初始状态
