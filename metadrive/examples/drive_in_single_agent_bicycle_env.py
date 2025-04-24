#!/usr/bin/env python
"""
테스트 스크립트:
- 우리가 만든 KinematicBicycleVehicle 클래스를 이용해,
  MetaDrive에서 키보드 조작(또는 AI 제어) 기반 주행을 시도해볼 수 있습니다.
- 스크립트 실행 시, H 키로 도움말을 볼 수 있고,
  W/A/S/D 조작으로 ego 차량(운동학 모델 기반) 이동을 테스트할 수 있습니다.
"""

import argparse
import logging
import random
import math
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import vehicle_type
from metadrive.constants import HELP_MESSAGE

#############################################################################
# 1) 우리가 만든 KinematicBicycleVehicle 클래스를 여기 직접 복붙하거나,
#    별도 파일에서 import 해옵니다.
#############################################################################

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.math import clip, wrap_to_pi, norm
from collections import deque


class KinematicBicycleVehicle(BaseVehicle):
    """
    KinematicBicycleVehicle 예시:
    - accel(m/s^2), steering_rate(rad/s)를 입력으로 받아
    - kinematic bicycle 식에 따라 (위치, 헤딩, 속도, 각속도) 갱신
    """

    def __init__(self,
                 vehicle_config=None,
                 name=None,
                 random_seed=None,
                 position=None,
                 heading=None,
                 _calling_reset=True):
        super().__init__(vehicle_config=vehicle_config,
                         name=name,
                         random_seed=random_seed,
                         position=position,
                         heading=heading,
                         _calling_reset=_calling_reset)
        # 조향각, 현재 속도
        self.steering_angle = 0.0
        self.current_speed = 0.0
        # 자전거 모델 휠베이스 (L). 여기서는 self.LENGTH를 사용하거나 따로 config에서 가져와도 됨
        self.wheelbase = self.LENGTH

    def before_step(self, action=None):
        """
        - action = [accel, steering_rate] (단위: m/s^2, rad/s)
        - dt = decision_repeat × physics_world_step_size
        - 자전거 모델 식:
            heading_rate = v * tan(steering_angle) / L
        """
        if action is None:
            action = [0.0, 0.0]  # accel=0, steer_rate=0

        accel = float(action[0])
        steering_rate = float(action[1])

        # dt 계산
        engine_config = self.engine.global_config
        decision_repeat = engine_config["decision_repeat"]
        physics_step = engine_config["physics_world_step_size"]
        dt = decision_repeat * physics_step

        # 스티어링각 적분
        self.steering_angle += steering_rate * dt
        # 최대 조향각(기존 self.max_steering는 deg단위)이므로 rad 변환
        max_steer_rad = float(self.max_steering) * math.pi / 180.0
        self.steering_angle = clip(self.steering_angle, -max_steer_rad,
                                   max_steer_rad)

        # 기존 헤딩, 속도
        old_heading = self.heading_theta
        old_speed = self.current_speed

        # 자전거 모델 요레이트
        heading_rate = old_speed * math.tan(
            self.steering_angle) / self.wheelbase
        # 속도 적분
        new_speed = old_speed + accel * dt
        if new_speed < 0.0:
            new_speed = 0.0

        # 헤딩 적분
        new_heading = old_heading + heading_rate * dt
        new_heading = wrap_to_pi(new_heading)

        # 위치 업데이트(구 헤딩 기준)
        old_x, old_y = self.position
        dx = old_speed * math.cos(old_heading) * dt
        dy = old_speed * math.sin(old_heading) * dt
        new_x = old_x + dx
        new_y = old_y + dy

        # 적용
        self.set_position((new_x, new_y))
        self.set_heading_theta(new_heading, in_rad=True)
        self.current_speed = new_speed

        # bullet엔진에도 속도/각속도 반영(시각적, 센서 반영용)
        heading_vec = np.array([math.cos(new_heading), math.sin(new_heading)])
        self.set_velocity(heading_vec, value=new_speed)
        self.set_angular_velocity(heading_rate, in_rad=True)

        # (선택) 바퀴 모델 돌리기
        # self.system.setSteeringValue(self.steering_angle, 0)
        # self.system.setSteeringValue(self.steering_angle, 1)

        return {
            "raw_action": (accel, steering_rate),
            "steering_angle_rad": self.steering_angle,
            "speed_m_s": self.current_speed,
            "heading_rate_rad_s": heading_rate,
        }

    def reset(self,
              vehicle_config=None,
              name=None,
              random_seed=None,
              position=None,
              heading=0.0,
              *args,
              **kwargs):
        # 상위 reset
        super().reset(vehicle_config=vehicle_config,
                      name=name,
                      random_seed=random_seed,
                      position=position,
                      heading=heading,
                      *args,
                      **kwargs)
        # 내부 변수 초기화
        self.steering_angle = 0.0
        self.current_speed = 0.0

        # Bullet 물리 이동을 끄고 싶다면:
        self.set_static(True)


#############################################################################
# 2) vehicle_type["k_bicycle"] = KinematicBicycleVehicle 로 등록
#############################################################################
vehicle_type["k_bicycle"] = KinematicBicycleVehicle
# bicycle_default


def main():
    """
    아래는 키보드 주행 스크립트.
    KinematicBicycleVehicle 를 ego 로 쓰기 위해,
    agent_configs.default_agent.vehicle_model="k_bicycle" 설정
    """
    config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_spawn_lane_index=False,
        random_lane_width=True,
        random_lane_num=True,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True,
                            show_navi_mark=False,
                            show_line_to_navi_mark=False),
        map=4,
        start_seed=10,
    )
    # 키보드와 이미지 관측 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation",
                        type=str,
                        default="lidar",
                        choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        from metadrive.component.sensors.rgb_camera import RGBCamera
        config.update(
            dict(image_observation=True,
                 sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
                 interface_panel=["rgb_camera", "dashboard"]))

    # 핵심: ego vehicle_model="k_bicycle" 지정
    # agent_configs = { "default_agent": {...} } 형태로
    config["agent_configs"] = {
        "default_agent":
            dict(
                vehicle_model="bicycle_default",  # 등록한 k_bicycle 키
            )
    }

    # env 생성
    from metadrive import MetaDriveEnv
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset(seed=21)
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True

        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("Observation keys:", o.keys())
        else:
            assert isinstance(o, np.ndarray)
            print("Observation shape:", o.shape)

        for i in range(1, 1000000000):
            # 여기서는 키보드 입력이 manual_control=True 옵션에 의해 자동으로 반영됨
            o, r, tm, tc, info = env.step([0, 0
                                          ])  # action=[0,0]는 무의미. 키보드가 override
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)":
                        "on" if env.current_track_agent.
                        expert_takeover else "off",
                    "Current Observation":
                        args.observation,
                    "Keyboard Control":
                        "W,A,S,D",
                })
            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)

            if (tm or tc) and info.get("arrive_dest", False):
                env.reset(env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()


if __name__ == "__main__":
    main()
