import torch
from torchvision import transforms
import numpy as np
import gym

from carla_env.wrappers import vector, get_displacement_vector

# 清空GPU缓存，释放内存
torch.cuda.empty_cache()


# 处理每一帧图像，将其转换为张量
def preprocess_frame(frame):
    # 定义图像预处理的转换流程，将图像转换为张量
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
    ])
    # 对帧进行预处理，并添加一个批次维度 (unsqueeze(0))
    frame = preprocess(frame).unsqueeze(0)
    return frame


# 创建一个函数，用于编码环境的状态信息
def create_encode_state_fn(measurements_to_include, CONFIG, vae=None):
    """
        measurements_to_include:  'state': ['steer', 'throttle', 'speed', 'waypoints', 'seg_camera'],
        返回一个函数，将环境的当前状态编码成特征向量。
    """

    # 根据是否包含特定测量数据，创建对应的标志位
    measure_flags = [
        "steer" in measurements_to_include,  # 是否包含转向角
        "throttle" in measurements_to_include,  # 是否包含油门值
        "speed" in measurements_to_include,  # 是否包含车速
        "angle_next_waypoint" in measurements_to_include,  # 是否包含下一个航点的角度
        "maneuver" in measurements_to_include,  # 是否包含当前道路机动信息
        "waypoints" in measurements_to_include,  # 是否包含接下来要经过的航点
        "rgb_camera" in measurements_to_include,  # 是否包含RGB相机图像数据
        "seg_camera" in measurements_to_include,  # 是否包含分割相机图像数据
        "end_wp_vector" in measurements_to_include,  # 是否包含到目标航点的位移向量
        "end_wp_fixed" in measurements_to_include,  # 是否包含到固定目标航点的位移向量
        "distance_goal" in measurements_to_include  # 是否包含目标距离
    ]

    # 定义一个函数，用来创建基于选择的测量数据的观测空间
    def create_observation_space():
        # 初始化一个空字典来存储观测空间
        # observation_space = {
        #   'vehicle_measures': [0.3, 0.5, 60, 1.2] --> 车辆的信息，如转向角，油门...
        #   'maneuver':   --> 道路机动类型，最多4种
        #   'waypoints':   --> 接下来要经过的航点， 15个航点，每个航点是2D坐标 --- 航点（Waypoints）可以理解为道路上的一个个“检查点”或“目标点”。车辆需要依次经过这些点
        #   'end_wp_vector':   --> 表示车辆与 动态目标航点 之间的相对位置
        #   'end_wp_fixed':   -->表示车辆与 固定目标航点 之间的相对位置
        #   'distance_goal':   -->  当前车辆距离目标航点的距离
        #   'rgb_camera'：  --> rgb相机的图像数据
        #   'seg_camera':   --> 分割相机的图像数据
        # }
        observation_space = {}
        low, high = [], []  # 定义状态值的下限和上限

        # 根据每个测量数据的标志位，设置其对应的状态空间范围
        if measure_flags[0]: low.append(-1), high.append(1)  # 转向角范围 (-1 到 1)
        if measure_flags[1]: low.append(0), high.append(1)  # 油门范围 (0 到 1)
        if measure_flags[2]: low.append(0), high.append(120)  # 车速范围 (0 到 120)
        if measure_flags[3]: low.append(-3.14), high.append(3.14)  # 角度范围 (-π 到 π)
        # 放的是车辆信息，并且限制了范围
        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        if measure_flags[4]: observation_space['maneuver'] = gym.spaces.Discrete(4)  # 道路机动类型为离散空间，最多4种类型
        if measure_flags[5]: observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2),
                                                                             dtype=np.float32)  # 15个航点，每个航点是2D坐标

        # 如果选择了图像数据（RGB或分割图像），设置相应的空间
        # 定义了RGB相机和分割相机，用于描述从这两个相机获取的图像数据的结构和取值范围
        if measure_flags[6]: observation_space['rgb_camera'] = gym.spaces.Box(low=0, high=255, shape=(
        CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        if measure_flags[7]: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(
        CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)

        # 如果选择了航点的位移向量或固定目标位移向量，设置相应的空间
        if measure_flags[8]: observation_space['end_wp_vector'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2),
                                                                                 dtype=np.float32)
        if measure_flags[9]: observation_space['end_wp_fixed'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2),
                                                                                dtype=np.float32)

        # 如果选择了目标距离，设置相应的空间
        if measure_flags[10]: observation_space['distance_goal'] = gym.spaces.Box(low=0, high=1500, shape=(1, 1),
                                                                                  dtype=np.float32)

        # 如果配置要求使用基于分割图的鸟瞰图（BEV），更新seg_camera空间大小
        if CONFIG.use_seg_bev: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(192, 192, 6),
                                                                                dtype=np.uint8)

        # 返回一个字典，表示所有可用观测数据的空间
        return gym.spaces.Dict(observation_space)

    # 定义编码函数，用于将环境的状态编码为特征向量
    def encode_state(env):
        # env --> 实际环境
        encoded_state = {}  # 创建一个字典来存储编码后的状态
        '''
        encoded_state = {
            'vehicle_measures':    ,
            'maneuver': 0,
            'waypoints': 0,
            'rgb_camera': 0,    
            'seg_camera': 0,
            'end_wp_vector': 0,
        }
        '''

        # 用于存储车辆相关度量的列表（转向角、油门、车速、下一个航点的角度）
        vehicle_measures = []
        if measure_flags[0]: vehicle_measures.append(env.vehicle.control.steer)  # 转向角
        if measure_flags[1]: vehicle_measures.append(env.vehicle.control.throttle)  # 油门
        if measure_flags[2]: vehicle_measures.append(env.vehicle.get_speed())  # 车速
        if measure_flags[3]: vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))  # 下一个航点的角度
        encoded_state['vehicle_measures'] = vehicle_measures  # 将车辆度量添加到状态中

        if measure_flags[4]: encoded_state['maneuver'] = env.current_road_maneuver.value  # 当前道路机动类型

        # 如果选择了航点，计算并添加相对的航点位置（位移向量）
        if measure_flags[5]:
            next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            vehicle_location = vector(env.vehicle.get_location())  # 获取车辆位置
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)  # 获取车辆的朝向角度

            # 计算每个航点相对车辆的位置（位移向量）
            relative_waypoints = np.zeros((15, 2))
            for i, w_location in enumerate(waypoints):
                relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]

            # 如果航点数量不足15个，通过参考最后两个航点之间的位移向量来推算剩余的航点
            if len(waypoints) < 15:
                start_index = len(waypoints)
                reference_vector = relative_waypoints[start_index - 1] - relative_waypoints[start_index - 2]
                for i in range(start_index, 15):
                    relative_waypoints[i] = relative_waypoints[i - 1] + reference_vector

            encoded_state['waypoints'] = relative_waypoints  # 将相对航点位置添加到状态中

        # 如果选择了RGB相机数据，添加图像数据
        if measure_flags[6]: encoded_state['rgb_camera'] = env.observation
        # 如果选择了分割相机数据，添加图像数据
        if measure_flags[7]: encoded_state['seg_camera'] = env.observation

        # 计算并添加到目标航点的位移向量
        if measure_flags[8]:
            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_vector'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]

        # 如果选择了固定目标航点的位移，计算并添加
        if measure_flags[9]:
            vehicle_location = vector(env.start_wp.transform.location)
            theta = np.deg2rad(env.start_wp.transform.rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_fixed'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]

        # 如果选择了目标距离，添加目标距离
        if measure_flags[10]:
            encoded_state['distance_goal'] = [[len(env.route_waypoints) - env.current_waypoint_index]]  # 距离目标的剩余航点数

        return encoded_state

    # 返回观测空间和状态编码函数
    return create_observation_space(), encode_state
