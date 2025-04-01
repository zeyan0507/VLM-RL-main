import os
import json
import cv2
import torch as th
from gym import spaces
from typing import Any, Dict, List, Union
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer, DictRolloutBuffer


class CLIPReplayBuffer(DictReplayBuffer):
    # 回放缓冲区用于存储智能体在训练过程中与环境的交互数据，包括观察、动作、奖励等信息，以便进行经验回放。
    def __init__(
            self,
            buffer_size: int,  # 回放缓冲区的大小
            observation_space: spaces.Space,  # 观察空间，描述环境返回的观察数据的空间结构
            action_space: spaces.Space,  # 动作空间，描述智能体可以采取的动作的空间结构
            device: Union[th.device, str] = "auto",  # 存储设备，默认自动选择
            n_envs: int = 1,  # 环境数量，支持多环境训练
            optimize_memory_usage: bool = False,  # 是否优化内存使用，默认为否
            handle_timeout_termination: bool = True,  # 是否处理超时终止，默认为是
    ):
        # 初始化父类的构造函数，并定义自定义的属性
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        # 自己修改：
        self.save_dir = "replay_buffer_data"  # 保存数据的文件夹
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  # 如果文件夹不存在，创建文件夹

        # 定义额外的列表，用于存储与 CLIP 奖励相关的额外信息
        self.render_arrays: List[np.ndarray] = []  # 存储渲染图像数据（例如，图像帧）
        self.base_rewards: List = []  # 存储基本奖励
        self.centering_factors: List = []  # 存储中心化因子
        self.angle_factors: List = []  # 存储角度因子
        self.speeds: List = []  # 存储速度信息
        self.distance_std_factors: List = []  # 存储距离标准差因子

    def add(
            self,
            obs: Dict[str, Any],  # 当前观察的数据
            next_obs: Dict[str, Any],  # 下一状态的观察数据
            action: np.ndarray,  # 当前执行的动作
            reward: np.ndarray,  # 当前步骤的奖励
            done: np.ndarray,  # 当前步骤是否结束
            infos: List[Dict[str, Any]],  # 环境返回的额外信息
    ) -> None:
        # 调用父类的 `add` 方法来添加经验到回放缓冲区
        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
        )

        # 确保不会超出缓冲区大小
        assert len(self.render_arrays) < self.buffer_size

        # 将额外的环境信息（例如渲染图像、速度、因子等）保存到相应的列表中
        self.render_arrays.append(infos[0]["render_array"])  # 渲染图像
        self.base_rewards.append(reward[0])  # 基础奖励
        self.centering_factors.append(infos[0]["centering_factor"])  # 中心化因子
        self.angle_factors.append(infos[0]["angle_factor"])  # 角度因子
        self.speeds.append(infos[0]["speed"])  # 速度信息
        self.distance_std_factors.append(infos[0]["distance_std_factor"])  # 距离标准差因子

    def clear_render_arrays(self) -> None:
        # 自己修改部分：
        # 在清除缓冲区数据之前，将数据保存到文件中
        self.save_data_to_file()

        # 清除缓冲区中与渲染图像相关的所有数据
        self.render_arrays = []
        self.base_rewards = []  # 清除基本奖励
        self.centering_factors = []  # 清除中心化因子
        self.angle_factors = []  # 清除角度因子
        self.speeds = []  # 清除速度信息
        self.distance_std_factors = []  # 清除距离标准差因子

    import json
    import numpy as np
    import cv2
    import os

    def save_data_to_file(self):
        # 保存图像数据和其他数据
        if len(self.render_arrays) > 0:
            # 假设保存图像为 PNG 文件，图像数据存在 render_arrays 中
            for i, frame in enumerate(self.render_arrays):
                img_filename = os.path.join(self.save_dir, f"frame_{i}.png")
                # 使用 OpenCV 保存图像
                cv2.imwrite(img_filename, frame)

        # 转换 numpy 数组为可序列化的类型
        def convert_to_serializable(data):
            if isinstance(data, np.ndarray):
                return data.tolist()  # Convert numpy arrays to lists
            elif isinstance(data, np.generic):  # Handle numpy scalar types like np.float32
                return data.item()
            return data

        # 保存其他数据（例如基础奖励、速度等）为 JSON 格式
        data = {
            "base_rewards": [convert_to_serializable(reward) for reward in self.base_rewards],
            "centering_factors": [convert_to_serializable(factor) for factor in self.centering_factors],
            "angle_factors": [convert_to_serializable(factor) for factor in self.angle_factors],
            "speeds": [convert_to_serializable(speed) for speed in self.speeds],
            "distance_std_factors": [convert_to_serializable(factor) for factor in self.distance_std_factors],
        }

        # 将数据保存为 JSON 文件
        json_filename = os.path.join(self.save_dir, "replay_buffer_data.json")
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file)

        print(f"Data saved to {self.save_dir}")


class CLIPRolloutBuffer(DictRolloutBuffer):
    def __init__(
            self,
            buffer_size: int,  # 缓冲区的大小
            observation_space: spaces.Space,  # 观察空间，描述观察数据的空间结构
            action_space: spaces.Space,  # 动作空间，描述动作的空间结构
            device: Union[th.device, str] = "auto",  # 存储设备，默认自动选择
            gae_lambda: float = 1,  # GAE的lambda参数
            gamma: float = 0.99,  # 折扣因子gamma
            n_envs: int = 1,  # 环境数量
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

        # 定义额外的列表，用于存储与 CLIP 奖励相关的额外信息
        self.render_arrays: List[np.ndarray] = []  # 存储渲染图像
        self.base_rewards: List = []  # 存储基础奖励
        self.centering_factors: List = []  # 存储中心化因子
        self.angle_factors: List = []  # 存储角度因子
        self.speeds: List = []  # 存储速度信息
        self.distance_std_factors: List = []  # 存储距离标准差因子

    def add(
            self,
            obs: Dict[str, np.ndarray],  # 当前观察数据
            action: np.ndarray,  # 当前动作
            reward: np.ndarray,  # 当前奖励
            episode_start: np.ndarray,  # 是否是回合的开始
            value: th.Tensor,  # 当前状态的值
            log_prob: th.Tensor,  # 当前动作的对数概率
            infos: List[Dict[str, Any]],  # 环境返回的额外信息
    ) -> None:
        super().add(
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob
        )

        # 确保不会超出缓冲区大小
        assert len(self.render_arrays) < self.buffer_size

        # 将额外的环境信息（例如渲染图像、速度、因子等）保存到相应的列表中
        self.render_arrays.append(infos[0]["render_array"])  # 渲染图像
        self.base_rewards.append(reward[0])  # 基础奖励
        self.centering_factors.append(infos[0]["centering_factor"])  # 中心化因子
        self.angle_factors.append(infos[0]["angle_factor"])  # 角度因子
        self.speeds.append(infos[0]["speed"])  # 速度信息
        self.distance_std_factors.append(infos[0]["distance_std_factor"])  # 距离标准差因子

    def clear_render_arrays(self) -> None:
        # 清除缓冲区中与渲染图像相关的所有数据
        self.render_arrays = []
        self.base_rewards = []  # 清除基本奖励
        self.centering_factors = []  # 清除中心化因子
        self.angle_factors = []  # 清除角度因子
        self.speeds = []  # 清除速度信息
        self.distance_std_factors = []  # 清除距离标准差因子
