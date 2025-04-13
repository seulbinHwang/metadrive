import gym
import numpy as np
from gym import spaces
from metadrive import MetaDriveEnv  # MetaDriveEnv가 기본 환경 클래스라고 가정


class MultiModalMetaDriveEnv(MetaDriveEnv):

    def __init__(self, config=None):
        super().__init__(config=config)
        # 각 모달리티에 대한 관측 공간 정의
        self.observation_space = spaces.Dict({
            'nav':
                spaces.Box(low=-np.inf,
                           high=np.inf,
                           shape=(25, 20, 12),
                           dtype=np.float32),
            'nav_mask':
                spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32),
            'agents':
                spaces.Box(low=-np.inf,
                           high=np.inf,
                           shape=(32, 21, 11),
                           dtype=np.float32),
            'agents_mask':
                spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),
            'lane':
                spaces.Box(low=-np.inf,
                           high=np.inf,
                           shape=(70, 20, 12),
                           dtype=np.float32),
            'lane_mask':
                spaces.Box(low=0, high=1, shape=(70,), dtype=np.float32),
            'static':
                spaces.Box(low=-np.inf,
                           high=np.inf,
                           shape=(5, 10),
                           dtype=np.float32),
            'static_mask':
                spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
        })

    def reset(self, **kwargs):
        original_obs = super().reset(**kwargs)
        return self._construct_multi_modal_obs()

    def step(self, action):
        original_obs, reward, done, info = super().step(action)
        return self._construct_multi_modal_obs(), reward, done, info

    def _construct_multi_modal_obs(self):
        # 1. Navigation 정보 (예: 웨이포인트 또는 경로 세그먼트)
        nav_data = np.zeros((25, 20, 12), dtype=np.float32)
        nav_mask = np.zeros(25, dtype=np.float32)
        # 최대 25개의 웨이포인트 또는 네비게이션 그리드 패치를 nav_data에 채움
        waypoints = self.vehicle.navigation.get_waypoint_list()  # 가상의 메소드
        for i, wp in enumerate(waypoints[:25]):
            nav_data[i] = extract_nav_patch(wp)  # 이 함수는 사용자가 정의한 20x12 특징 추출 함수
            nav_mask[i] = 1.0

        # 2. 주변 에이전트들의 궤적
        agents_data = np.zeros((32, 21, 11), dtype=np.float32)
        agents_mask = np.zeros(32, dtype=np.float32)
        neighbor_vehicles = self.engine.get_nearby_objects(
            self.vehicle, object_type="vehicle", N=32)
        for j, veh in enumerate(neighbor_vehicles):
            traj = get_future_trajectory(
                veh, horizon=21)  # 예: 해당 차량의 21 스텝 예측 또는 샘플링된 궤적
            agents_data[j] = traj[:21]  # 각 traj 스텝은 11개의 특징 (위치, 속도 등)을 가짐
            agents_mask[j] = 1.0

        # 3. 차선 정보
        lane_data = np.zeros((70, 20, 12), dtype=np.float32)
        lane_mask = np.zeros(70, dtype=np.float32)
        lane_segments = self.engine.get_map().get_surrounding_lanes(
            self.vehicle, radius=50, max_segments=70)
        for k, lane_seg in enumerate(lane_segments):
            lane_data[k] = extract_lane_patch(
                lane_seg)  # 차선 구간을 20x12 특징(예: 로컬 그리드)로 표현
            lane_mask[k] = 1.0

        # 4. 정적 객체 정보
        static_data = np.zeros((5, 10), dtype=np.float32)
        static_mask = np.zeros(5, dtype=np.float32)
        static_objs = self.engine.get_nearby_objects(self.vehicle,
                                                     object_type="static",
                                                     N=5)
        for m, obj in enumerate(static_objs):
            static_data[m] = encode_static_object(
                obj)  # 정적 객체를 10개의 특징(유형, 위치 등)으로 인코딩
            static_mask[m] = 1.0

        # dict로 결합
        obs = {
            'nav': nav_data,
            'nav_mask': nav_mask,
            'agents': agents_data,
            'agents_mask': agents_mask,
            'lane': lane_data,
            'lane_mask': lane_mask,
            'static': static_data,
            'static_mask': static_mask
        }
        return obs
