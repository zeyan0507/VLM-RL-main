import torch as th
from box import Box  # 用于处理嵌套字典类型的对象
from stable_baselines3.common.noise import NormalActionNoise  # 导入正常分布的动作噪声（用于DDPG等算法）
import numpy as np
from utils import lr_schedule  # 导入自定义学习率调度函数

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # 导入特征提取器的基类
from stable_baselines3.common.preprocessing import get_flattened_obs_dim  # 获取扁平化观测空间维度的函数
import torch.nn as nn  # 导入PyTorch的神经网络模块
import gymnasium as gym  # 导入Gymnasium库，包含各种强化学习环境
import torch  # 导入PyTorch


# 自定义卷积神经网络（CNN）类，用于从图像输入中提取特征
class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=1):
        super(CustomCNN, self).__init__()
        n_input_channels = input_shape[0]

        # 如果输入图像的通道数是3（RGB图像）
        if n_input_channels == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),  # (16, 58, 38)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 28, 18)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 13, 8)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (128, 6, 4)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 2)
                nn.ReLU(),
                nn.Flatten(),  # 展平多维输出
            )
        else:  # 如果不是RGB图像（例如单通道图像）
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),  # 展平多维输出
            )

        # 使用一个零张量来计算卷积层输出的形状
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).view(-1).shape[0]

        # 定义全连接层，用于输出特征
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)  # 卷积层处理
        x = self.linear(x)  # 全连接层处理
        return x


# 自定义多输入特征提取器，用于处理包含多个不同类型输入的环境
class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        extractors = {}  # 存储各个子空间的特征提取器
        total_concat_size = 0  # 存储最终拼接后的特征维度

        # 如果观测空间是字典类型（例如包含多个不同的传感器数据）
        if isinstance(observation_space, gym.spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if key == "seg_camera":  # 如果是分割图像数据，使用自定义CNN提取器
                    extractors[key] = CustomCNN(subspace.shape, features_dim=features_dim)
                    total_concat_size += features_dim
                else:  # 其他类型的空间使用扁平化处理
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)
        else:  # 如果只有一个输入（例如单一图像）
            extractors["default"] = CustomCNN(observation_space.shape, features_dim=features_dim)
            total_concat_size = features_dim

        # 将所有子空间的特征提取器存储在ModuleDict中
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size  # 总特征维度

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []  # 用于存储每个输入的特征表示

        # 如果输入是字典（多种类型的输入）
        if isinstance(observations, dict):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))  # 提取每个子空间的特征
        else:
            encoded_tensor_list.append(self.extractors["default"](observations))  # 单一输入时提取特征
        return torch.cat(encoded_tensor_list, dim=1)  # 将所有特征拼接在一起

# 定义不同强化学习算法的超参数设置
algorithm_params = {
    "PPO": dict(
        device="cuda:0",  # 使用GPU
        learning_rate=lr_schedule(1e-4, 1e-6, 2),  # 学习率调度
        gamma=0.98,  # 折扣因子
        gae_lambda=0.95,  # GAE的lambda参数
        clip_range=0.2,  # PPO算法中的裁剪范围
        ent_coef=0.05,  # 熵正则化系数
        n_epochs=10,  # 每次更新的epoch数
        n_steps=512,  # 每次采样的步数 --> 1024 -->512
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,  # 激活函数
            net_arch=[dict(pi=[500, 300], vf=[500, 300])],  # 网络架构
            features_extractor_class=CustomMultiInputExtractor,  # 使用自定义特征提取器
            features_extractor_kwargs=dict(features_dim=256),
        )
    ),
    "SAC": dict(
        device="cuda:0",  # 使用GPU
        learning_rate=lr_schedule(5e-4, 1e-6, 2),  # 学习率调度
        buffer_size=10000,  # 缓冲区大小
        batch_size=128,  # 批次大小 --- 256 --> 128
        ent_coef='auto',  # 自动调整熵系数
        gamma=0.98,  # 折扣因子
        tau=0.02,  # 软更新系数
        train_freq=64,  # 每训练多少次
        gradient_steps=64,  # 每次更新的梯度步骤数
        learning_starts=10000,  # 开始学习的步骤数
        use_sde=True,  # 使用SDE（Stochastic Differential Equations）
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),  # 策略网络的架构
    ),
    "DDPG": dict(
        device="cuda:0",  # 使用GPU
        gamma=0.98,  # 折扣因子
        buffer_size=200000,  # 缓冲区大小
        learning_starts=10000,  # 学习开始的步骤数
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),  # 添加噪声
        gradient_steps=-1,  # 每次更新的梯度步骤数
        learning_rate=lr_schedule(5e-4, 1e-6, 2),  # 学习率调度
        policy_kwargs=dict(net_arch=[400, 300]),  # 策略网络的架构
    ),
    "SAC_CLIP": dict(
        device="cuda:0",  # 使用GPU
        learning_rate=lr_schedule(1e-4, 5e-7, 2),  # 学习率调度
        buffer_size=10000,  # 缓冲区大小
        batch_size=128,  # 批次大小 --- 256 --> 128
        ent_coef='auto',  # 自动调整熵系数
        gamma=0.98,  # 折扣因子
        tau=0.02,  # 软更新系数
        train_freq=64,  # 每训练多少次
        gradient_steps=64,  # 每次更新的梯度步骤数
        learning_starts=10000,  # 开始学习的步骤数
        use_sde=True,  # 使用SDE
        policy_kwargs=dict(
            log_std_init=-3, net_arch=[500, 300],
            features_extractor_class=CustomMultiInputExtractor,  # 使用自定义特征提取器
            features_extractor_kwargs=dict(features_dim=256),
        )
    ),
}


# 定义不同状态空间的设置
states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"],
    "5": ["steer", "throttle", "speed", "waypoints", "seg_camera"],  # 状态空间5包含分割图像
}
reward_params = {
    # reward_fn_5_default: 默认的奖励函数配置
    "reward_fn_5_default": dict(
        early_stop=True,  # 是否启用早期停止，如果达到某些条件即提前终止
        min_speed=20.0,  # 最低速度，单位：km/h
        max_speed=35.0,  # 最高速度，单位：km/h
        target_speed=25.0,  # 目标速度，单位：km/h
        max_distance=3.0,  # 从中心位置的最大允许距离，超出此距离将终止
        max_std_center_lane=0.4,  # 中心车道偏差的最大标准差
        max_angle_center_lane=90,  # 车道偏离的最大角度，单位：度
        penalty_reward=-10,  # 奖励函数中的惩罚奖励，负值表示惩罚
    ),

    # reward_fn_5_no_early_stop: 不启用早期停止的奖励函数配置
    "reward_fn_5_no_early_stop": dict(
        early_stop=False,  # 不启用早期停止
        min_speed=20.0,  # 最低速度，单位：km/h
        max_speed=35.0,  # 最高速度，单位：km/h
        target_speed=25.0,  # 目标速度，单位：km/h
        max_distance=3.0,  # 从中心位置的最大允许距离，超出此距离将终止
        max_std_center_lane=0.4,  # 中心车道偏差的最大标准差
        max_angle_center_lane=90,  # 车道偏离的最大角度，单位：度
        penalty_reward=-10,  # 奖励函数中的惩罚奖励，负值表示惩罚
    ),

    # reward_fn_5_best: 最佳奖励函数配置
    "reward_fn_5_best": dict(
        early_stop=True,  # 启用早期停止
        min_speed=20.0,  # 最低速度，单位：km/h
        max_speed=35.0,  # 最高速度，单位：km/h
        target_speed=25.0,  # 目标速度，单位：km/h
        max_distance=2.0,  # 从中心位置的最大允许距离，超出此距离将终止
        max_std_center_lane=0.35,  # 中心车道偏差的最大标准差
        max_angle_center_lane=90,  # 车道偏离的最大角度，单位：度
        penalty_reward=-10,  # 奖励函数中的惩罚奖励，负值表示惩罚
    ),

    # reward_clg: 用于车辆碰撞检测的奖励函数配置
    "reward_clg": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",  # 预训练模型
        batch_size=64,  # 每批次处理的样本数
        target_prompts=[  # 目标提示，模型根据这些提示生成对应的结果
            "Two cars have collided with each other on the road",  # 两辆车发生碰撞
            "The road is clear with no car accidents",  # 路面没有车祸
        ],
    ),

    # reward_lord: 用于车辆碰撞检测的奖励函数配置（仅一个目标）
    "reward_lord": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",  # 预训练模型
        batch_size=64,  # 每批次处理的样本数
        target_prompts=[  # 目标提示，模型根据这些提示生成对应的结果
            "Two cars have collided with each other on the road",  # 两辆车发生碰撞
        ],
    ),

    # reward_vlm_rm: 用于安全驾驶检测的奖励函数配置
    "reward_vlm_rm": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",  # 预训练模型
        batch_size=64,  # 每批次处理的样本数
        alpha=0.5,  # 权重系数，用于平衡不同的奖励
        target_prompts=[  # 目标提示，模型根据这些提示生成对应的结果
            "A car is driving safely",  # 一辆车在安全驾驶
        ],
        baseline_prompts=[  # 基准提示，用于对比
            "A car",  # 一辆车
        ],
    ),

    # reward_fn_Chen: 另一个奖励函数配置，带有较大的最大距离
    "reward_fn_Chen": dict(
        early_stop=True,  # 启用早期停止
        min_speed=0.0,  # 最低速度，单位：km/h
        max_speed=28.8,  # 最高速度，单位：km/h
        target_speed=25.0,  # 目标速度，单位：km/h
        max_distance=4.0,  # 从中心位置的最大允许距离，超出此距离将终止
        max_std_center_lane=0.4,  # 中心车道偏差的最大标准差
        max_angle_center_lane=90,  # 车道偏离的最大角度，单位：度
        penalty_reward=-10,  # 奖励函数中的惩罚奖励，负值表示惩罚
    ),

    # reward_fn_ASAP: 适用于更高速度范围的奖励函数配置
    "reward_fn_ASAP": dict(
        early_stop=True,  # 启用早期停止
        min_speed=0.0,  # 最低速度，单位：km/h
        max_speed=50.0,  # 最高速度，单位：km/h
        target_speed=30.0,  # 目标速度，单位：km/h
        max_distance=3.0,  # 从中心位置的最大允许距离，超出此距离将终止
        max_std_center_lane=0.4,  # 中心车道偏差的最大标准差
        max_angle_center_lane=90,  # 车道偏离的最大角度，单位：度
        penalty_reward=-5,  # 奖励函数中的惩罚奖励，负值表示惩罚
    ),
}

# 配置PPO算法，包含特定的参数和奖励函数
_CONFIG_1 = {
    "algorithm": "PPO",  # 使用的算法是Proximal Policy Optimization（PPO）
    "algorithm_params": algorithm_params["PPO"],  # PPO算法的特定参数
    "state": states["5"],  # 使用的状态配置，'5'代表特定的状态设置
    "action_smoothing": 0.75,  # 动作平滑因子，控制动作的平滑度
    "reward_fn": "reward_fn5",  # 使用的奖励函数，命名为'reward_fn5'
    "reward_params": reward_params["reward_fn_5_default"],  # 奖励函数的参数
    "obs_res": (80, 120),  # 观测分辨率，设置为(高度, 宽度)
    "seed": 100,  # 随机种子，确保结果的可重复性
    "wrappers": [],  # 包装器的列表，用于修改环境或代理行为，此处为空
    "use_rgb_bev": False,  # 是否使用RGB鸟瞰图（BEV），此处为False
}

# 配置SAC算法，包含特定的参数和奖励函数
_CONFIG_2 = {
    "algorithm": "SAC",  # 使用的算法是Soft Actor-Critic（SAC）
    "algorithm_params": algorithm_params["SAC"],  # SAC算法的特定参数
    "state": states["5"],  # 使用的状态配置，和_CONFIG_1相同
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数，和_CONFIG_1相同
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置CLIP-SAC算法，使用VLM-RL奖励类型
_CONFIG_vlm_rl = {
    "algorithm": "CLIP-SAC",  # 使用的算法是CLIP-SAC，结合了SAC和CLIP模型
    "algorithm_params": algorithm_params["SAC_CLIP"],  # SAC-CLIP特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_clg"],  # CLIP特定的奖励参数
    "vlm_reward_type": "VLM-RL",  # 奖励类型为VLM-RL
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声配置
    "use_seg_bev": True,  # 使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置CLIP-PPO算法，使用VLM-RL奖励类型
_CONFIG_vlm_rl_ppo = {
    "algorithm": "CLIP-PPO",  # 使用的算法是CLIP-PPO，结合了PPO和CLIP模型
    "algorithm_params": algorithm_params["PPO"],  # PPO特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_clg"],  # CLIP特定的奖励参数
    "vlm_reward_type": "VLM-RL",  # 奖励类型为VLM-RL
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声
    "action_space_type": "discrete",  # 动作空间为离散型
    "use_seg_bev": True,  # 使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置CLIP-SAC算法，使用LORD奖励类型
_CONFIG_lord = {
    "algorithm": "CLIP-SAC",  # 使用的算法是CLIP-SAC
    "algorithm_params": algorithm_params["SAC_CLIP"],  # SAC-CLIP特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_lord"],  # LORD特定的奖励参数
    "vlm_reward_type": "LORD",  # 奖励类型为LORD
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声
    "use_seg_bev": False,  # 不使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置CLIP-SAC算法，使用LORD-Speed奖励类型
_CONFIG_lord_speed = {
    "algorithm": "CLIP-SAC",  # 使用的算法是CLIP-SAC
    "algorithm_params": algorithm_params["SAC_CLIP"],  # SAC-CLIP特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_lord"],  # LORD特定的奖励参数
    "vlm_reward_type": "LORD-Speed",  # 奖励类型为LORD-Speed
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声
    "use_seg_bev": False,  # 不使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置CLIP-SAC算法，使用VLM-RM奖励类型
_CONFIG_vlm_rm = {
    "algorithm": "CLIP-SAC",  # 使用的算法是CLIP-SAC
    "algorithm_params": algorithm_params["SAC_CLIP"],  # SAC-CLIP特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_vlm_rm"],  # VLM-RM特定的奖励参数
    "vlm_reward_type": "VLM-RM",  # 奖励类型为VLM-RM
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声
    "use_seg_bev": False,  # 不使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置CLIP-SAC算法，使用VLM-SR奖励类型
_CONFIG_vlm_sr = {
    "algorithm": "CLIP-SAC",  # 使用的算法是CLIP-SAC
    "algorithm_params": algorithm_params["SAC_CLIP"],  # SAC-CLIP特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_vlm_rm"],  # VLM-RM特定的奖励参数
    "vlm_reward_type": "VLM-SR",  # 奖励类型为VLM-SR
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声
    "use_seg_bev": False,  # 不使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置CLIP-SAC算法，使用RoboCLIP奖励类型
_CONFIG_roboclip = {
    "algorithm": "CLIP-SAC",  # 使用的算法是CLIP-SAC
    "algorithm_params": algorithm_params["SAC_CLIP"],  # SAC-CLIP特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn5",  # 使用的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "clip_reward_params": reward_params["reward_vlm_rm"],  # VLM-RM特定的奖励参数
    "vlm_reward_type": "RoboCLIP",  # 奖励类型为RoboCLIP
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "action_noise": {},  # 空的动作噪声
    "use_seg_bev": False,  # 不使用分段鸟瞰图（BEV）
    "use_rgb_bev": True,  # 使用RGB鸟瞰图（BEV）
}

# 配置TIRL算法与SAC
_CONFIG_tirl_sac = {
    "algorithm": "SAC",  # SAC算法
    "algorithm_params": algorithm_params["SAC"],  # SAC特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_simple",  # 简单的奖励函数，用于TIRL
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置TIRL算法与PPO
_CONFIG_tirl_ppo = {
    "algorithm": "PPO",  # PPO算法
    "algorithm_params": algorithm_params["PPO"],  # PPO特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_simple",  # 简单的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置ChatScene与SAC
_CONFIG_chatscene_sac = {
    "algorithm": "SAC",  # SAC算法
    "algorithm_params": algorithm_params["SAC"],  # SAC特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_chatscene",  # ChatScene特定的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置ChatScene与PPO
_CONFIG_chatscene_ppo = {
    "algorithm": "PPO",  # PPO算法
    "algorithm_params": algorithm_params["PPO"],  # PPO特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_chatscene",  # ChatScene奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置Revolve与SAC
_CONFIG_revolve = {
    "algorithm": "SAC",  # SAC算法
    "algorithm_params": algorithm_params["SAC"],  # SAC特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_revolve",  # Revolve特定的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置Revolve自动奖励函数与SAC
_CONFIG_revolve_auto = {
    "algorithm": "SAC",  # SAC算法
    "algorithm_params": algorithm_params["SAC"],  # SAC特定的参数
    "state": states["5"],  # 使用的状态配置
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_revolve_auto",  # 自动Revolve特定的奖励函数
    "reward_params": reward_params["reward_fn_5_default"],  # 默认奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 100,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置Chen与SAC
_CONFIG_Chen = {
    "algorithm": "SAC",  # SAC算法
    "algorithm_params": algorithm_params["SAC"],  # SAC特定的参数
    "state": states["5"],  # 使用的状态配置
    "vae_model": None,  # 没有VAE模型
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_Chen",  # Chen特定的奖励函数
    "reward_params": reward_params["reward_fn_Chen"],  # Chen特定的奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 120,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置ASAP与PPO
_CONFIG_ASAP = {
    "algorithm": "PPO",  # PPO算法
    "algorithm_params": algorithm_params["PPO"],  # PPO特定的参数
    "state": states["5"],  # 使用的状态配置
    "vae_model": None,  # 没有VAE模型
    "action_smoothing": 0.75,  # 动作平滑因子
    "reward_fn": "reward_fn_ASAP",  # ASAP特定的奖励函数
    "reward_params": reward_params["reward_fn_ASAP"],  # ASAP特定的奖励函数参数
    "obs_res": (80, 120),  # 观测分辨率
    "seed": 120,  # 随机种子
    "wrappers": [],  # 空的包装器列表
    "use_rgb_bev": False,  # 不使用RGB BEV
}

# 配置字典，包含所有配置项
CONFIGS = {
    "1": _CONFIG_1,
    "2": _CONFIG_2,
    "vlm_rl": _CONFIG_vlm_rl,
    "vlm_rl_ppo": _CONFIG_vlm_rl_ppo,
    "lord": _CONFIG_lord,
    "lord_speed": _CONFIG_lord_speed,
    "vlm_rm": _CONFIG_vlm_rm,
    "vlm_sr": _CONFIG_vlm_sr,
    "roboclip": _CONFIG_roboclip,
    "tirl_sac": _CONFIG_tirl_sac,
    "tirl_ppo": _CONFIG_tirl_ppo,
    "chatscene_sac": _CONFIG_chatscene_sac,
    "chatscene_ppo": _CONFIG_chatscene_ppo,
    "revolve": _CONFIG_revolve,
    "revolve_auto": _CONFIG_revolve_auto,
    "Chen": _CONFIG_Chen,
    "ASAP": _CONFIG_ASAP,
}

# 全局变量CONFIG，用于存储当前选中的配置
CONFIG = None


# 设置配置函数，根据配置名称选择并返回配置 -- vlm-rl
def set_config(config_name):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)  # 设置全局变量CONFIG为选中的配置
    return CONFIG  # 返回选中的配置
