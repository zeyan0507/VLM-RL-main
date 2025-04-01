from typing import List, Tuple

import open_clip
import torch
import torch.nn as nn
from torch import Tensor

from clip.transform import image_transform


# CLIPEmbed 是一个封装 CLIP 模型图像编码的类
class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model  # CLIP 模型
        # 获取 CLIP 模型的图像输入尺寸
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        # 使用图像转换函数来处理图像输入
        self.transform = image_transform(image_size)

    @torch.inference_mode()  # 关闭梯度计算，节省内存
    def forward(self, x):
        # 如果输入图像的通道数不为 3（即非 RGB），则调整维度顺序
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        # 禁用梯度计算，并在支持的情况下开启自动混合精度计算
        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            # 对图像进行转换，调整为 CLIP 模型所需的输入尺寸
            x = self.transform(x)  # [batch, 3, 244, 244]
            # 使用 CLIP 模型编码图像，输出图像嵌入
            x = self.clip_model.encode_image(x, normalize=True)  # [batch, 1024]
        return x


# CLIPReward 是计算与 CLIP 模型相关的奖励的类
class CLIPReward(nn.Module):
    def __init__(
            self,
            *,
            model: CLIPEmbed,
            alpha: float,
            target_prompts: torch.Tensor,
            baseline_prompts: torch.Tensor,
    ) -> None:
        super().__init__()
        self.clip_embed_module = model  # 引入 CLIPEmbed 实例
        # 对目标提示（prompts）进行编码（生成嵌入向量）
        targets = self.embed_prompts(target_prompts)
        # 将目标嵌入向量注册为模型的缓冲区
        self.register_buffer("targets", targets)

        if len(baseline_prompts) > 0:
            # 如果有基准提示，则对基准提示进行编码
            baselines = self.embed_prompts(baseline_prompts)
            # 计算目标与基准的差异（用于后续计算投影矩阵）
            direction = targets - baselines  # [1, 1024]
            # 将基准嵌入和差异向量注册为模型的缓冲区
            self.register_buffer("baselines", baselines)
            self.register_buffer("direction", direction)
            self.alpha = alpha  # alpha 超参数，用于计算投影
            # 计算投影矩阵
            projection = self.compute_projection(alpha, self.direction)  # [1024, 1024]
            # 将投影矩阵注册为模型的缓冲区
            self.register_buffer("projection", projection)

    def compute_projection(self, alpha: float, direction) -> torch.Tensor:
        """
        计算投影矩阵，用于将嵌入向量映射到目标向量的方向
        投影矩阵是根据目标和基准提示之间的方向差异计算的
        """
        projection = direction.T @ direction / torch.norm(direction) ** 2
        # 创建单位矩阵（对角线为 1）
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        # 通过混合投影矩阵和单位矩阵，控制平滑度
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, vlm_reward_type: str) -> torch.Tensor:
        """
        根据奖励类型计算相应的奖励值
        """
        if vlm_reward_type == "VLM-RL":
            return self.forward_vlm_rl(x)
        elif "LORD" in vlm_reward_type:
            return self.forward_lord(x)
        elif vlm_reward_type == "VLM-RM":
            return self.forward_vlm_rm(x)
        elif vlm_reward_type in ["VLM-SR", "RoboCLIP"]:
            return self.forward_vlm_sr(x)
        else:
            raise NotImplementedError  # 未实现的奖励类型

    @torch.inference_mode()
    def forward_vlm_rl(self, x: torch.Tensor) -> torch.Tensor:
        """
        VLM-RL 奖励计算：计算输入向量与目标向量之间的相似度差异
        """
        x = x / torch.norm(x, dim=-1, keepdim=True)  # 归一化输入
        y = (x @ self.targets.T)  # 计算输入向量与目标向量的相似度 :点积
        z = y[:, 0] - y[:, 1]  # 计算目标的差异
        return z

    @torch.inference_mode()
    def forward_lord(self, x: torch.Tensor) -> torch.Tensor:
        """
        LORD 奖励计算：计算输入向量与目标向量的相似度差异
        """
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        z = 1 - y  # 计算相似度与 1 之间的差异
        return z.squeeze()  # 去掉多余的维度

    @torch.inference_mode()
    def forward_vlm_rm(self, x: torch.Tensor) -> torch.Tensor:
        """
        VLM-RM 奖励计算：计算输入与目标向量之间的距离
        """
        x = x / torch.norm(x, dim=-1, keepdim=True)
        # 使用投影矩阵计算输入和目标向量之间的距离
        y = 1 - (torch.norm((x - self.targets) @ self.projection, dim=-1) ** 2) / 2
        return y

    @torch.inference_mode()
    def forward_vlm_sr(self, x: torch.Tensor) -> torch.Tensor:
        """
        VLM-SR 或 RoboCLIP 奖励计算：计算输入向量与目标向量的相似度
        """
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        return y.squeeze()  # 去掉多余的维度

    @torch.inference_mode()
    def get_pos_neg(self, x: torch.Tensor):
        """
        获取正负提示的相似度值
        """
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        return y[:, 0], y[:, 1]  # 返回正负提示的相似度值

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """
        将提示列表转换为 CLIP 模型可接受的文本输入（tokenization）
        """
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """
        将提示嵌入为向量
        """
        with torch.no_grad():
            x = self.clip_embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)  # 归一化嵌入向量
        return x

    def embed_images(self, x):
        """
        将图像输入嵌入为向量
        """
        return self.clip_embed_module.forward(x)


# 计算奖励的函数
def compute_rewards(
        model: CLIPReward,
        frames: torch.Tensor,
        batch_size: int,
        vlm_reward_type: str = "VLM-RL",
) -> Tensor:
    """
    计算每一帧图像的奖励值，返回奖励向量
    """
    # 确保输入的设备是 CPU
    assert frames.device == torch.device("cpu")
    n_samples = len(frames)
    # 初始化一个奖励向量
    basic_rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()  # 设置模型为评估模式，关闭 Dropout 等操作
    with torch.no_grad():
        # 按批次计算奖励
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i: i + batch_size].to(next(model.parameters()).device)
            with torch.no_grad():
                embeddings = model.clip_embed_module(frames_batch)  # 获取图像的嵌入向量
                rewards_batch = model(embeddings, vlm_reward_type=vlm_reward_type)  # 计算奖励
            basic_rewards[i: i + batch_size] = rewards_batch

    return basic_rewards  # 返回计算得到的奖励值
