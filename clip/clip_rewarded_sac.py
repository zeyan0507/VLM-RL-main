import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any

import numpy as np
import open_clip
import stable_baselines3.common.noise as sb3_noise
import torch
from box import Box
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from clip.clip_buffer import CLIPReplayBuffer
from clip.clip_reward_model import compute_rewards, CLIPEmbed, CLIPReward

from config import CONFIG

# SelfCLIPRewardedSAC 是一个类型变量，用于引用当前类 CLIPRewardedSAC，以便进行类型推导。
SelfCLIPRewardedSAC = TypeVar("SelfCLIPRewardedSAC", bound="CLIPRewardedSAC")


class CLIPRewardedSAC(SAC):
    # 该类继承了 SAC 类，意味着它基于 Stable-Baselines3 中的 SAC 算法。
    replay_buffer: CLIPReplayBuffer  # 使用 CLIPReplayBuffer 作为回放缓冲区； 用来存储历史经验。

    def __init__(
        self,
        *,
        env: VecEnv,  # env 是训练的环境，类型是 VecEnv
        config: Box,  # config 是一个包含算法配置的对象。
        inference_only: bool = False,  # 是否仅用于推理（默认为False，表示训练模式）
    ):
        self.config = config  # 保存配置

        # 处理环境中的动作噪声，增强探索 ---训练的时候应该是none
        if config.action_noise:
            mean = config.action_noise.mean * np.ones(env.action_space.shape)
            sigma = config.action_noise.sigma * np.ones(env.action_space.shape)
            if config.action_noise.name == "NormalActionNoise":
                action_noise = sb3_noise.NormalActionNoise(mean=mean, sigma=sigma)
            elif config.action_noise.name == "OrnsteinUhlenbeckActionNoise":
                action_noise = sb3_noise.OrnsteinUhlenbeckActionNoise(
                    mean=mean,
                    sigma=sigma,
                    theta=config.action_noise.theta,
                    dt=config.action_noise.dt,
                )
            else:
                raise ValueError(
                    f"Unknown action noise name: {config.action_noise.name}"
                )
        else:
            action_noise = None  # 如果没有配置噪声，则不使用

        # 调用父类的构造函数进行初始化
        super().__init__(
            env=env,
            policy='MultiInputPolicy',  # 多输入策略
            replay_buffer_class=CLIPReplayBuffer,  # 使用自定义的 CLIP 回放缓存
            # CLIPRolloutBuffer 用于在每一步的执行过程中，保存每个时刻的观察、奖励、动作等信息。
            # 这些类的作用是为强化学习模型提供一个缓冲区（Replay Buffer），用于存储智能体在环境中与环境交互时的经验。
            # 这些经验包括观察、动作、奖励等信息。通过这些经验，模型能够进行回放学习，从而提升性能。
            tensorboard_log='tensorboard',  # 日志保存路径
            seed=config.seed,  # 随机种子 -- 100
            action_noise=action_noise,  # 动作噪声 -- None
            **self.config.algorithm_params,  # 从配置中传入算法的其他参数
        )
        self.ep_clip_info_buffer = None  # 用于存储关于每个回合的 CLIP 信息
        self.inference_only = inference_only  # 是否仅进行推理模式
        if not self.inference_only: # 默认的是False，也就是训练模式，也就是需要执行下面的代码
            self._load_modules()  # 加载模型模块
            self.previous_num_timesteps = 0  # 上一个时间步
            self.previous_num_episodes = 0  # 上一个回合数

    def _dump_logs(self) -> None:
        pass  # 此处不保存日志

    def _load_modules(self):
        # 加载 CLIP 模型并初始化奖励模型
        # 获取 CLIP 模型名称的前缀和预训练模型的路径
        # model_name_prefix：'ViT-bigG-14'
        # pretrained：'laion2b_s39b_b160k'
        model_name_prefix, pretrained = self.config.clip_reward_params.pretrained_model.split("/")

        clip_model = open_clip.create_model(
            model_name=model_name_prefix, pretrained=pretrained  # 加载预训练的 CLIP 模型
        )# 加载预训练模型 pretrained='laion2b_s39b_b160k' 表示加载 ViT-bigG-14 模型的预训练权重。

        clip_model = CLIPEmbed(clip_model)  # 用 CLIPEmbed 包装 CLIP 模型
        target_prompts = CLIPReward.tokenize_prompts(self.config.clip_reward_params.target_prompts)  # 目标提示词
        baseline_prompts = CLIPReward.tokenize_prompts(self.config.clip_reward_params.baseline_prompts)  # 基线提示词

        # 创建 CLIPReward 模型
        clip_model = CLIPReward(
            model=clip_model,
            alpha=self.config.clip_reward_params.alpha,  # 奖励模型的超参数
            target_prompts=target_prompts,
            baseline_prompts=baseline_prompts, # 这里好像设置为空了
        )
        self.reward_model = clip_model.eval().to(self.device)  # 将模型转为评估模式并放到设备上

    def _compute_clip_rewards(self):
        # CLIPReward 是自定义的类，用来计算奖励。CLIP 模型将根据图像与文本之间的对比学习来进行训练和推理。
        assert self.env is not None  # 确保环境存在

        replay_buffer_pos = self.replay_buffer.pos  # 回放缓存的位置
        total_timesteps = self.num_timesteps - self.previous_num_timesteps  # 总时间步数
        env_episode_timesteps = total_timesteps // self.env.num_envs  # 每个环境的时间步数

        # 渲染帧（render frame）通常是指通过仿真或环境交互所生成的图像帧
        frames = torch.from_numpy(np.array(self.replay_buffer.render_arrays))  # 从回放缓存中获取渲染帧
        width = frames.shape[2]  # 帧的宽度
        left = width // 2  # 取右半部分
        frames = frames[:, :, left:, :]  # 只使用图像的右半部分

        # 计算复杂的奖励
        rewards0 = compute_rewards(
            #这个模型会根据图像 (frames) 和文本提示的相似度来计算奖励，
            # 具体的计算方式会依赖于 reward_model 和 vlm_reward_type。
            model=self.reward_model,
            frames=frames,
            batch_size=self.config.clip_reward_params.batch_size,
            vlm_reward_type=self.config.vlm_reward_type,
        )
        rewards0 = rewards0.numpy().reshape(-1, 1)  # 将奖励转换为数组
        print("Clip reward ...")
        print(list(np.round(rewards0.flatten(), 4)))  # 打印计算出来的奖励

        base_reward = np.array(self.replay_buffer.base_rewards).reshape(-1, 1)  # 基础奖励
        speeds = np.array(self.replay_buffer.speeds)  # 速度信息
        print("Speed ...")
        print(list(np.round(speeds.flatten(), 4)))  # 打印速度

        # 根据不同的奖励类型调整奖励
        if self.config.vlm_reward_type == "VLM-RL":
            thre_min, thre_max = 0.0, 0.03
            rewards0 = np.clip(rewards0, a_min=thre_min, a_max=thre_max)  # 将奖励值限制在范围内
            rewards0 = 1 - (rewards0 - thre_min) / (thre_max - thre_min)  # 标准化奖励

            centering_factors = np.array(self.replay_buffer.centering_factors)
            angle_factors = np.array(self.replay_buffer.angle_factors)
            distance_std_factors = np.array(self.replay_buffer.distance_std_factors)
            desired_speed = np.clip(rewards0.flatten(), 0.0, 1.0) * CONFIG.reward_params.target_speed
            r_speeds = 1.0 - np.abs(speeds - desired_speed) / CONFIG.reward_params.target_speed
            # 综合奖励函数的实现
            rewards = (r_speeds * centering_factors * angle_factors * distance_std_factors).reshape(-1, 1)
            rewards = np.where(base_reward < 0, base_reward, rewards)  # 如果基础奖励为负，则使用基础奖励

        elif self.config.vlm_reward_type == "LORD-Speed":
            lord_speed_r = 1.0 - np.abs(speeds - 20.0) / 20.0
            rewards = rewards0 + lord_speed_r.reshape(-1, 1)  # 根据速度调整奖励
        elif self.config.vlm_reward_type == "VLM-SR":
            rewards = np.where(rewards0 > 0.32, 1, -1)  # 对奖励进行二值化处理
        else:
            rewards = rewards0  # 如果没有特别的奖励类型，则直接使用计算出的奖励

        print("Final reward ...")
        print(list(np.round(rewards.flatten(), 4)))  # 打印最终的奖励
        self.replay_buffer.clear_render_arrays()  # 清除回放缓存中的渲染数组

        # 如果不是推理模式，更新回放缓冲区中的奖励
        if not self.inference_only:
            if replay_buffer_pos - env_episode_timesteps >= 0:
                self.replay_buffer.rewards[
                replay_buffer_pos - env_episode_timesteps: replay_buffer_pos
                ] = rewards
            else:
                self.replay_buffer.rewards[
                -(env_episode_timesteps - replay_buffer_pos):
                ] = rewards[: env_episode_timesteps - replay_buffer_pos]
                self.replay_buffer.rewards[:replay_buffer_pos] = rewards[
                                                                 env_episode_timesteps - replay_buffer_pos:
                                                                 ]

    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        # 收集回合数据
        rollout = super().collect_rollouts(*args, **kwargs)
        if not self.inference_only:
            self._compute_clip_rewards()  # 计算 CLIP 奖励
            self.previous_num_timesteps = self.num_timesteps  # 更新时间步
            self.previous_num_episodes = self._episode_num  # 更新回合数
        return rollout

    def _log(self) -> None:
        # 记录训练过程中的一些统计信息
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_gt_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # 训练方法
        self._log()  # 记录日志
        super().train(gradient_steps, batch_size)  # 调用父类的训练方法

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        *args,
    ) -> Tuple[int, BaseCallback]:
        # 设置学习过程的相关参数
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            *args,
        )
        if self.ep_clip_info_buffer is None or reset_num_timesteps:
            self.ep_clip_info_buffer = deque(maxlen=100)
        return total_timesteps, callback

    def learn(self: SelfCLIPRewardedSAC, *args, **kwargs) -> SelfCLIPRewardedSAC:
        # 学习方法，调用父类的学习函数
        assert not self.inference_only  # 确保不是推理模式
        self.previous_num_timesteps = 0  # 重置时间步
        self.previous_num_episodes = 0  # 重置回合数
        return super().learn(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:  # type: ignore
        # 保存模型
        super().save(*args, exclude=["reward_model", "worker_frames_tensor"], **kwargs)

    @classmethod
    def load(
            cls: Type[SelfCLIPRewardedSAC],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            load_clip: bool = True,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfCLIPRewardedSAC:
        # 加载模型
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
        )
        assert data is not None, "No data found in the saved file"  # 确保数据存在
        assert params is not None, "No params found in the saved file"  # 确保参数存在
        # 后续代码用于加载模型的状态和参数，并恢复训练环境
