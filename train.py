# 导入必要的库
import warnings
import os
from datetime import datetime

# 忽略警告信息
warnings.filterwarnings("ignore")
# 设置 TensorFlow 的日志级别为3，即仅显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 导入必要的模块
import argparse
import config

# 命令行输入： python train.py --config=vlm_rl
# --start_carla --no_render --total_timesteps=1_000_000
# --port=2000 --device=cuda:0
# 使用vlm-rl配置文件；训练总时间步数-1000000
# 设置命令行参数 训练的配置文件为 tirl_sac
parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")  # 设置服务器IP
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")  # 设置端口号
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timestep to train for")  # 设置训练的总步数
parser.add_argument("--start_carla", action="store_true", help="If True, start a CARLA server")  # 是否启动 CARLA 服务器
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")  # 是否渲染环境
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")  # 设置渲染环境的帧率
parser.add_argument("--num_checkpoints", type=int, default=100, help="Checkpoint number")  # 设置保存模型的检查点频率
parser.add_argument("--log_dir", type=str, default="tensorboard", help="Directory to save logs")  # 设置日志保存目录
parser.add_argument("--device", type=str, default="cuda:0", help="cpu, cuda:0, cuda:1, cuda:2")  # 设置计算设备
# parser.add_argument("--config", type=str, default="vlm_rl_ppo", help="Config to use (default: vlm_rl)")  # 设置配置文件名称
parser.add_argument("--config", type=str, default="vlm_rl", help="Config to use (default: vlm_rl)")  # 设置配置文件名称


# 解析命令行参数
args = vars(parser.parse_args())
# print(args)

# 设置配置
CONFIG = config.set_config(args["config"])  # 从配置文件加载设置 -- vlm_rl --在config中配置了对应的
print("CONFIG信息格式：\n",CONFIG)
print("-------------------------")
# CONFIG:
# {'algorithm': 'CLIP-SAC', 'algorithm_params':
#         {'device': 'cuda:0', 'learning_rate': <function lr_schedule.<locals>.func at 0x0000025B867B65E0>,
#         'buffer_size': 10000, 'batch_size': 128, 'ent_coef': 'auto', 'gamma': 0.98,
#         'tau': 0.02, 'train_freq': 64, 'gradient_steps': 64, 'learning_starts': 10000, 'use_sde': True,
#         'policy_kwargs': {'log_std_init': -3, 'net_arch': [500, 300], 'features_extractor_class': <class 'config.CustomMultiInputExtractor'>,
#         'features_extractor_kwargs': {'features_dim': 256}}},
# 'state': ['steer', 'throttle', 'speed', 'waypoints', 'seg_camera'],
# 'action_smoothing': 0.75, 'reward_fn': 'reward_fn5',
# 'reward_params': {'early_stop': True, 'min_speed': 20.0, 'max_speed': 35.0, 'target_speed': 25.0,
#                           'max_distance': 3.0, 'max_std_center_lane': 0.4, 'max_angle_center_lane': 90, 'penalty_reward': -10},
# 'clip_reward_params': {'pretrained_model': 'ViT-bigG-14/laion2b_s39b_b160k', 'batch_size': 64,
#                        'target_prompts': ['Two cars have collided with each other on the road', 'The road is clear with no car accidents']},
# 'vlm_reward_type': 'VLM-RL', 'obs_res': (80, 120), 'seed': 100, 'wrappers': [], 'action_noise': {}, 'use_seg_bev': True, 'use_rgb_bev': True}

# print(CONFIG)
CONFIG.algorithm_params.device = args["device"]  # 设置计算设备 -- GPU

# 导入强化学习算法和所需的回调函数
from stable_baselines3 import PPO, DDPG, SAC
from clip.clip_rewarded_sac import CLIPRewardedSAC
from clip.clip_rewarded_ppo import CLIPRewardedPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class

# 创建日志目录
os.makedirs(args["log_dir"], exist_ok=True)

# 定义算法字典，将算法名称与相应的算法类关联
algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC,
                  "CLIP-SAC": CLIPRewardedSAC, "CLIP-PPO": CLIPRewardedPPO}
# 如果配置中的算法不存在于字典中，则抛出错误
if CONFIG.algorithm not in algorithm_dict:
    raise ValueError("Invalid algorithm name")
# print(CONFIG.algorithm) # CLIP-SAC

# 获取所选算法的类
AlgorithmRL = algorithm_dict[CONFIG.algorithm] # CLIPRewardedSAC
# print(CONFIG)
# 创建环境的观测空间和状态编码函数
# observation_space--定义一个存储数据的空间,包括车辆的运动信息以及环境信息，也就是还没存数据
# 而，encode_state --> 就是存入数据的  不过这里只是定义了函数
observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
# 根据配置选择动作空间类型
action_space_type = 'continuous' if CONFIG.action_space_type != 'discrete' else 'discrete' # 默认就是连续的啦

# 创建CARLA环境 --carla_route_env
# print(CONFIG.reward_fn) --reward_fn5 奖励函数
# obs_res = (80,120)
# 构建了个自动驾驶的仿真环境
env = CarlaRouteEnv(obs_res=CONFIG.obs_res, host=args["host"], port=args["port"],
                    reward_fn=reward_functions[CONFIG.reward_fn], observation_space=observation_space,
                    encode_state_fn=encode_state_fn, fps=args["fps"],
                    action_smoothing=CONFIG.action_smoothing, action_space_type=action_space_type,
                    activate_spectator=args["no_render"], activate_render=args["no_render"],
                    activate_bev=CONFIG.use_rgb_bev, activate_seg_bev=CONFIG.use_seg_bev,
                    activate_traffic_flow=True, start_carla=args["start_carla"],
                    )

# 对环境进行包装，如果配置中指定了包装类 -- 应该没有
for wrapper_class_str in CONFIG.wrappers:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)

# 根据算法类型选择合适的模型 CLIPRewardedSAC --设置了奖励模型
if AlgorithmRL.__name__ == "CLIPRewardedSAC":
    model = CLIPRewardedSAC(env=env, config=CONFIG)
elif AlgorithmRL.__name__ == "CLIPRewardedPPO":
    model = CLIPRewardedPPO(env=env, config=CONFIG)
else:
    model = AlgorithmRL('MultiInputPolicy', env, verbose=1, seed=CONFIG.seed, tensorboard_log=args["log_dir"],
                        **CONFIG.algorithm_params)

# 设置模型保存路径和名称 ---tensorboard
model_suffix = "{}_id{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), args['config'])
model_name = f'{model.__class__.__name__}_{model_suffix}'
model_dir = os.path.join(args["log_dir"], model_name)
print("保存的文件名：\n",model_dir)
print("-------------------------------")
# 配置日志记录
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# 保存配置文件
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

# 开始训练
model.learn(total_timesteps=args["total_timesteps"],
            callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
                save_freq=args["total_timesteps"] // args["num_checkpoints"],
                save_path=model_dir, name_prefix="model")], reset_num_timesteps=False)
