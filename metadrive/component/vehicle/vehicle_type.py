import platform

from panda3d.core import LineSegs, NodePath
from panda3d.core import Material, Vec3, LVecBase4

from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.constants import Semantics
from metadrive.engine.asset_loader import AssetLoader

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.engine.logger import get_logger

logger = get_logger()

from collections import deque
import math
import numpy as np

# NuPlan (example) imports
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint, StateSE2, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

# MetaDrive imports
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils.math import wrap_to_pi

from metadrive.utils.math import clip, wrap_to_pi


class DefaultVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.DEFAULT_VEHICLE)
    # LENGTH = 4.51
    # WIDTH = 1.852
    # HEIGHT = 1.19
    TIRE_RADIUS = 0.313
    TIRE_WIDTH = 0.25
    MASS = 1100
    LATERAL_TIRE_TO_CENTER = 0.815
    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    path = ('ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)
           )  # asset path, scale, offset, HPR

    DEFAULT_LENGTH = 4.515  # meters
    DEFAULT_HEIGHT = 1.19  # meters
    DEFAULT_WIDTH = 1.852  # meters

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH


class HistoryDefaultVehicle(DefaultVehicle):
    """
    - DefaultVehicle를 상속받아, 매 스텝마다 EgoState를 생성/저장하는 기능을 추가한 클래스입니다.
    - ego_history: Deque[EgoState] 로 관리.
    """

    def __init__(self,
                 vehicle_config=None,
                 name=None,
                 random_seed=None,
                 position=None,
                 heading=None,
                 _calling_reset=True,
                 ego_history_maxlen=100):
        """
        :param ego_history_maxlen: ego_history에 최대 몇 개의 EgoState를 저장할지
        """
        super().__init__(vehicle_config=vehicle_config,
                         name=name,
                         random_seed=random_seed,
                         position=position,
                         heading=heading,
                         _calling_reset=_calling_reset)

        # EgoState 기록용 덱
        self.ego_history = deque(maxlen=ego_history_maxlen)

        # -------------------------------
        #  (1) NuPlan VehicleParameters 생성
        # -------------------------------
        #  DefaultVehicle에서:
        #     self.LENGTH = 4.515 (ex)
        #     self.WIDTH  = 1.852
        #     self.HEIGHT = 1.19
        #
        #  또한:
        #     self.FRONT_WHEELBASE = 1.05234
        #     self.REAR_WHEELBASE  = 1.4166
        #     -> wheel_base = FRONT_WHEELBASE + REAR_WHEELBASE = 2.46894
        #
        #  "front_length" = rear axle ~ front bumper distance
        #  "rear_length"  = rear axle ~ rear bumper distance
        #
        #   (차 중점을 origin이라 가정시)
        #   - center ~ front bumper: LENGTH/2 = 2.2575
        #   - center ~ rear bumper : LENGTH/2 = 2.2575
        #   - center ~ rear axle   : REAR_WHEELBASE=1.4166 (뒤쪽)
        #
        #   => rear axle ~ front bumper = (1.4166 + 2.2575) = 3.6741
        #   => rear axle ~ rear bumper  = (2.2575 - 1.4166) = 0.8409
        #
        #  "cog_position_from_rear_axle" 는 rear axle ~ COG 거리
        #   여기서는 "COG=차 중심"이라 가정 -> 1.4166 m
        #
        #  vehicle_name = self.name
        #  vehicle_type = "MetaDrive" (등 적절히)
        #
        # -------------------------------
        total_length = self.LENGTH  # 4.515
        half_len = total_length / 2.0  # 2.2575
        front_axle_dist = self.FRONT_WHEELBASE  # 1.05234
        rear_axle_dist = self.REAR_WHEELBASE  # 1.4166

        # wheel_base
        wheel_base_val = front_axle_dist + rear_axle_dist  # 2.46894

        # rear axle -> front bumper
        front_length_val = rear_axle_dist + half_len  # 1.4166 + 2.2575 = 3.6741
        # rear axle -> rear bumper
        rear_length_val = abs(half_len - rear_axle_dist)  # 0.8409

        # cog_position_from_rear_axle
        cog_from_rear = rear_axle_dist  # 1.4166

        self._nuplan_vehicle_params = VehicleParameters(
            width=self.WIDTH,  # 1.852
            front_length=front_length_val,
            rear_length=rear_length_val,
            cog_position_from_rear_axle=cog_from_rear,
            wheel_base=wheel_base_val,
            vehicle_name=self.name if self.name else "EgoVehicle",
            vehicle_type="MetaDrive",
            height=self.HEIGHT  # 1.19
        )

        # 이전 스텝 속도(배속) 등을 저장하여 가속도 계산 용도 (option)
        # TODO: 이 부분이 수정 되어야 할 수도 있음
        self._previous_velocity = None
        self._previous_ang_vel = None

        self._previous_time_s = 0.0

    @property
    def ego_state(self):
        """
        현재까지 기록된 EgoState를 반환합니다.
        :return: EgoState
        """
        return self.ego_history[-1] if len(self.ego_history) > 0 else None

    def reset(self, *args, **kwargs):
        """에피소드 시작시 호출 – 기록 초기화"""
        super().reset(*args, **kwargs)
        self.ego_history.clear()

        # 2) EgoState 생성
        current_ego_state = self._create_ego_state()
        # 3) 덱에 추가
        self.ego_history.append(current_ego_state)

    def after_step(self):
        """
        MetaDrive에서 매 스텝 후에 불리는 메서드.
        여기서 NuPlan의 EgoState를 생성하고, ego_history 덱에 쌓아둡니다.
        """
        # 1) 기존 DefaultVehicle의 after_step 수행
        step_info = super().after_step()

        # 2) EgoState 생성
        current_ego_state = self._create_ego_state()
        # 3) 덱에 추가
        self.ego_history.append(current_ego_state)

        return step_info

    def _create_ego_state(self) -> EgoState:
        """
        DefaultVehicle 상태 -> NuPlan EgoState 변환
        """
        # 1) 시뮬레이션 시간 (초 -> 마이크로초)
        """
        TODO: 
`self.engine.global_config["physics_world_step_size"] * 
self.engine.global_config["decision_repeat"]` 의 배수

reset 되면 time_us가 0으로 초기화 되는지 확인
        """
        step_index = self.engine.episode_step

        # engine.global_config에서 dt 가져오기
        # dt = physics_world_step_size * decision_repeat
        dt = (self.engine.global_config["physics_world_step_size"] *
              self.engine.global_config["decision_repeat"])

        # step_index * dt => 현재까지 진행된 시뮬레이션 시간 [s]
        current_time_s = step_index * dt

        # float(s) -> 마이크로초로 변환
        time_us = int(current_time_s * 1e6)
        time_point = TimePoint(time_us)

        dt = current_time_s - self._previous_time_s

        # 2) 차량 중심 좌표를 StateSE2 로 생성
        #    (DefaultVehicle.position 은 (x, y), heading_theta는 라디안)
        x_c = float(self.position[0])
        y_c = float(self.position[1])
        yaw_c = wrap_to_pi(self.heading_theta)  # 라디안, -π ~ +π
        center_pose = StateSE2(x_c, y_c, yaw_c)

        # 3) 속도(중심) / 가속도(중심)
        #    - self.velocity => (vx, vy) in m/s (world coords)
        vx = float(self.velocity[0])
        vy = float(self.velocity[1])
        center_velocity_2d = StateVector2D(vx, vy)

        #    간단한 finite difference로 a = (v - v_prev) / dt
        if self._previous_velocity is None:
            center_acc_2d = StateVector2D(0.0, 0.0)
        else:
            v_prev = self._previous_velocity
            ax = (vx - v_prev[0]) / dt
            ay = (vy - v_prev[1]) / dt
            center_acc_2d = StateVector2D(ax, ay)

        # 4) 타이어 조향각 (radians)
        #    self.steering: -1 ~ +1
        #    max_steering=40 (deg) => 약 0.698 rad
        max_steer_deg = self.max_steering  # default=40 deg
        max_steer_rad = math.radians(max_steer_deg)
        tire_angle = float(self.steering) * max_steer_rad

        # 5) BulletVehicle의 각속도(회전)는 rad/s, Z축에 해당
        #    -> (ang_vel_z = self.body.getAngularVelocity()[2])  # (ZUp)
        #    (기본적으로 z-index가 2지만, 여기서는 -1일 수도 있음. 실제 값 확인 필요)
        #    일단 Panda3D에서 ZUp이므로 getAngularVelocity()[-1] 사용
        ang_vel_z = self.body.getAngularVelocity()[-1]  # float, rad/s
        #    각가속도도 finite diff
        if self._previous_ang_vel is None:
            ang_acc_z = 0.0
        else:
            prev_ang_vel = self._previous_ang_vel
            ang_acc_z = (ang_vel_z - prev_ang_vel) / dt

        # 6) EgoState 생성
        ego_state = EgoState.build_from_center(
            center=center_pose,
            center_velocity_2d=center_velocity_2d,
            center_acceleration_2d=center_acc_2d,
            tire_steering_angle=tire_angle,
            time_point=time_point,
            vehicle_parameters=self._nuplan_vehicle_params,
            is_in_auto_mode=True,
            angular_vel=ang_vel_z,  # [rad/s]
            angular_accel=ang_acc_z,  # [rad/s²]
        )

        # 기록 갱신
        self._previous_velocity = np.array([vx, vy], dtype=float)
        self._previous_time_s = current_time_s
        self._previous_ang_vel = ang_vel_z

        return ego_state


class KinematicBicycleVehicle(HistoryDefaultVehicle):
    """
    A custom vehicle class implementing a simple Kinematic Bicycle Model.
    Action format: [acceleration, steering_rate]
      - acceleration: (float) desired linear acceleration in m/s^2
      - steering_rate: (float) desired steering angular velocity in rad/s
    We manually integrate position, heading, speed, etc. each step
    and override the Bullet physical force-based approach.
    """
    PARAMETER_SPACE = ParameterSpace(
        VehicleParameterSpace.KINEMATIC_BICYCLE_VEHICLE)
    TIRE_RADIUS = 0.313
    TIRE_WIDTH = 0.25
    MASS = 1100
    LATERAL_TIRE_TO_CENTER = 1.1485
    FRONT_WHEELBASE = 1.419
    REAR_WHEELBASE = 1.67
    path = ('ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)
           )  # asset path, scale, offset, HPR

    DEFAULT_LENGTH = 4.049 + 1.127  # meters
    DEFAULT_HEIGHT = 1.777  # meters
    DEFAULT_WIDTH = 1.1485 * 2.  # meters

    def __init__(self,
                 vehicle_config=None,
                 name=None,
                 random_seed=None,
                 position=None,
                 heading=None):
        super().__init__(
            vehicle_config=vehicle_config,
            name=name,
            random_seed=random_seed,
            position=position,
            heading=heading,
            _calling_reset=False  # we'll manually call reset() below
        )
        self._motion_model = KinematicBicycleModel(
            self._nuplan_vehicle_params,
            max_steering_angle=self.max_steering * math.pi / 180.0)
        # Initialize internal states for kinematic model

    def reset(self,
              vehicle_config=None,
              name=None,
              random_seed=None,
              position=None,
              heading=0.0,
              *args,
              **kwargs):
        """
        Called when resetting the vehicle. We'll also set it static so bullet won't move it.
        """
        # Use parent's reset to do basic config, set position, etc.
        super().reset(vehicle_config=vehicle_config,
                      name=name,
                      random_seed=random_seed,
                      position=position,
                      heading=heading,
                      *args,
                      **kwargs)

        # Let the bullet engine see this as static, so it won't update via forces
        self.set_static(True)
        # Initialize states

    def before_step(self, action=None):
        """
        This function is called each simulation step before the physics engine updates.
        We'll override the default bullet-based logic with a kinematic bicycle step.
        """
        # Skip parent's before_step so we don't invoke `_set_action` from Bullet
        # We can still do some data recording:
        step_info = {"raw_action": action}

        if action is None:
            # if no action given, assume zero accel / zero steering_rate
            action = [0.0, 0.0]
        else:
            # typically, env might clip it to [-1,1], but we can interpret in real physical range
            pass
        # parse user action
        accel_m_s2 = float(action[0])  # linear acceleration
        steering_rate = float(
            action[1])  # how fast steering angle changes (rad/s)
        dynamic_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=self.ego_state.car_footprint.
            rear_axle_to_center_dist,
            rear_axle_velocity_2d=self.ego_state.dynamic_car_state.
            rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(accel_m_s2, 0),
            tire_steering_rate=steering_rate,
        )
        dt = (self.engine.global_config["physics_world_step_size"] *
              self.engine.global_config["decision_repeat"])
        sampling_time = TimePoint(int(dt * 1e6))
        current_state: EgoState = self._motion_model.propagate_state(
            state=self.ego_state,
            ideal_dynamic_state=dynamic_state,
            sampling_time=sampling_time)
        """
        I have to call(to update) below state & methods using the current_state
        state
            self.steering : steering
                self._set_action 으로 먼저 해보고, 이상하면 바꾸자.
        
        methods
            self.set_position : x, y
            self.set_heading_theta : theta
            self.set_velocity : velocity
            self.set_angular_velocity : 
            
        """
        steering = current_state.tire_steering_angle
        self._set_action([steering, 0.0])  # TODO: check
        self.set_position(current_state.center.point.array)
        self.set_heading_theta(current_state.center.heading)
        self.set_velocity(
            current_state.dynamic_car_state.center_velocity_2d.array)
        self.set_angular_velocity(
            current_state.dynamic_car_state.angular_velocity)
        # If we want to store or return any step info:
        return step_info

    def get_state(self):
        """
        Override for convenience: include our custom kinematic states in the dictionary.
        """
        base_state = super().get_state()

        return base_state


# When using DefaultVehicle as traffic, please use this class.


class TrafficDefaultVehicle(DefaultVehicle):
    pass


class StaticDefaultVehicle(DefaultVehicle):
    PARAMETER_SPACE = ParameterSpace(
        VehicleParameterSpace.STATIC_DEFAULT_VEHICLE)


class XLVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.XL_VEHICLE)
    # LENGTH = 5.8
    # WIDTH = 2.3
    # HEIGHT = 2.8
    TIRE_RADIUS = 0.37
    TIRE_MODEL_CORRECT = -1
    REAR_WHEELBASE = 1.075
    FRONT_WHEELBASE = 1.726
    LATERAL_TIRE_TO_CENTER = 0.931
    CHASSIS_TO_WHEEL_AXIS = 0.3
    TIRE_WIDTH = 0.5
    MASS = 1600
    LIGHT_POSITION = (-0.75, 2.7, 0.2)
    SEMANTIC_LABEL = Semantics.TRUCK.label
    path = ('truck/vehicle.gltf', (1, 1, 1), (0, 0.25, 0.04), (0, 0, 0))

    DEFAULT_LENGTH = 5.74  # meters
    DEFAULT_HEIGHT = 2.8  # meters
    DEFAULT_WIDTH = 2.3  # meters

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH


class LVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.L_VEHICLE)
    # LENGTH = 4.5
    # WIDTH = 1.86
    # HEIGHT = 1.85
    TIRE_RADIUS = 0.429
    REAR_WHEELBASE = 1.218261
    FRONT_WHEELBASE = 1.5301
    LATERAL_TIRE_TO_CENTER = 0.75
    TIRE_WIDTH = 0.35
    MASS = 1300
    LIGHT_POSITION = (-0.65, 2.13, 0.3)
    DEFAULT_LENGTH = 4.87  # meters
    DEFAULT_HEIGHT = 1.85  # meters
    DEFAULT_WIDTH = 2.046  # meters

    path = ['lada/vehicle.gltf', (1.1, 1.1, 1.1), (0, -0.27, 0.07), (0, 0, 0)]

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH


class MVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
    # LENGTH = 4.4
    # WIDTH = 1.85
    # HEIGHT = 1.37
    TIRE_RADIUS = 0.39
    REAR_WHEELBASE = 1.203
    FRONT_WHEELBASE = 1.285
    LATERAL_TIRE_TO_CENTER = 0.803
    TIRE_WIDTH = 0.3
    MASS = 1200
    LIGHT_POSITION = (-0.67, 1.86, 0.22)
    DEFAULT_LENGTH = 4.6  # meters
    DEFAULT_HEIGHT = 1.37  # meters
    DEFAULT_WIDTH = 1.85  # meters
    path = ['130/vehicle.gltf', (1, 1, 1), (0, -0.05, 0.1), (0, 0, 0)]

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH


class SVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.S_VEHICLE)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    LATERAL_TIRE_TO_CENTER = 0.7
    TIRE_TWO_SIDED = True
    FRONT_WHEELBASE = 1.385
    REAR_WHEELBASE = 1.11
    TIRE_RADIUS = 0.376
    TIRE_WIDTH = 0.25
    MASS = 800
    LIGHT_POSITION = (-0.57, 1.86, 0.23)
    DEFAULT_LENGTH = 4.3  # meters
    DEFAULT_HEIGHT = 1.7  # meters
    DEFAULT_WIDTH = 1.7  # meters

    @property
    def path(self):
        if self.use_render_pipeline and platform.system() != "Linux":
            # vfs = VirtualFileSystem.get_global_ptr()
            # vfs.mount(convert_path(AssetLoader.file_path("models", "beetle")), "/$$beetle_model", 0)
            return [
                'beetle/vehicle.bam', (0.0077, 0.0077, 0.0077),
                (0.04512, -0.24 - 0.04512, 1.77), (-90, -90, 0)
            ]
        else:
            factor = 1
            return [
                'beetle/vehicle.gltf', (factor, factor, factor),
                (0, -0.2, 0.03), (0, 0, 0)
            ]

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH


class VaryingDynamicsVehicle(DefaultVehicle):

    @property
    def WIDTH(self):
        return self.config["width"] if self.config[
            "width"] is not None else super(VaryingDynamicsVehicle, self).WIDTH

    @property
    def LENGTH(self):
        return self.config[
            "length"] if self.config["length"] is not None else super(
                VaryingDynamicsVehicle, self).LENGTH

    @property
    def HEIGHT(self):
        return self.config[
            "height"] if self.config["height"] is not None else super(
                VaryingDynamicsVehicle, self).HEIGHT

    @property
    def MASS(self):
        return self.config["mass"] if self.config[
            "mass"] is not None else super(VaryingDynamicsVehicle, self).MASS

    def reset(
            self,
            random_seed=None,
            vehicle_config=None,
            position=None,
            heading: float = 0.0,  # In degree!
            *args,
            **kwargs):

        assert "width" not in self.PARAMETER_SPACE
        assert "height" not in self.PARAMETER_SPACE
        assert "length" not in self.PARAMETER_SPACE
        should_force_reset = False
        if vehicle_config is not None:
            if vehicle_config["width"] is not None and vehicle_config[
                    "width"] != self.WIDTH:
                should_force_reset = True
            if vehicle_config["height"] is not None and vehicle_config[
                    "height"] != self.HEIGHT:
                should_force_reset = True
            if vehicle_config["length"] is not None and vehicle_config[
                    "length"] != self.LENGTH:
                should_force_reset = True
            if "max_engine_force" in vehicle_config and \
                vehicle_config["max_engine_force"] is not None and \
                vehicle_config["max_engine_force"] != self.config["max_engine_force"]:
                should_force_reset = True
            if "max_brake_force" in vehicle_config and \
                vehicle_config["max_brake_force"] is not None and \
                vehicle_config["max_brake_force"] != self.config["max_brake_force"]:
                should_force_reset = True
            if "wheel_friction" in vehicle_config and \
                vehicle_config["wheel_friction"] is not None and \
                vehicle_config["wheel_friction"] != self.config["wheel_friction"]:
                should_force_reset = True
            if "max_steering" in vehicle_config and \
                vehicle_config["max_steering"] is not None and \
                vehicle_config["max_steering"] != self.config["max_steering"]:
                self.max_steering = vehicle_config["max_steering"]
                should_force_reset = True
            if "mass" in vehicle_config and \
                vehicle_config["mass"] is not None and \
                vehicle_config["mass"] != self.config["mass"]:
                should_force_reset = True

        # def process_memory():
        #     import psutil
        #     import os
        #     process = psutil.Process(os.getpid())
        #     mem_info = process.memory_info()
        #     return mem_info.rss
        #
        # cm = process_memory()

        if should_force_reset:
            self.destroy()
            self.__init__(vehicle_config=vehicle_config,
                          name=self.name,
                          random_seed=self.random_seed,
                          position=position,
                          heading=heading,
                          _calling_reset=False)

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format("1 Force Re-Init Vehicle", (lm - cm) / 1e6))
            # cm = lm

        assert self.max_steering == self.config["max_steering"]

        ret = super(VaryingDynamicsVehicle,
                    self).reset(random_seed=random_seed,
                                vehicle_config=vehicle_config,
                                position=position,
                                heading=heading,
                                *args,
                                **kwargs)

        # lm = process_memory()
        # print("{}:  Reset! Mem Change {:.3f}MB".format("2 Force Reset Vehicle", (lm - cm) / 1e6))
        # cm = lm

        return ret


class VaryingDynamicsBoundingBoxVehicle(VaryingDynamicsVehicle):

    def __init__(self,
                 vehicle_config: dict = None,
                 name: str = None,
                 random_seed=None,
                 position=None,
                 heading=None,
                 **kwargs):

        # TODO(pzh): The above code is removed for now. How we get BUS label?
        #  vehicle_config has 'width' 'length' and 'height'
        # if vehicle_config["width"] < 0.0:
        #     self.SEMANTIC_LABEL = Semantics.CAR.label
        # else:
        #     self.SEMANTIC_LABEL = Semantics.BUS.label

        super(VaryingDynamicsBoundingBoxVehicle,
              self).__init__(vehicle_config=vehicle_config,
                             name=name,
                             random_seed=random_seed,
                             position=position,
                             heading=heading,
                             **kwargs)

    def _add_visualization(self):
        if self.render:
            path, scale, offset, HPR = self.path

            # PZH: Note that we do not use model_collection as a buffer here.
            # if path not in BaseVehicle.model_collection:

            # PZH: Load a box model and resize it to the vehicle size
            car_model = AssetLoader.loader.loadModel(
                AssetLoader.file_path("models", "box.bam"))

            car_model.setTwoSided(False)
            BaseVehicle.model_collection[path] = car_model
            car_model.setScale((self.WIDTH, self.LENGTH, self.HEIGHT))
            # car_model.setZ(-self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS + self.HEIGHT / 2)
            car_model.setZ(0)
            # model default, face to y
            car_model.setHpr(*HPR)
            car_model.instanceTo(self.origin)

            show_contour = self.config[
                "show_contour"] if "show_contour" in self.config else False
            if show_contour:
                # ========== Draw the contour of the bounding box ==========
                # Draw the bottom of the car first
                line_seg = LineSegs("bounding_box_contour1")
                zoffset = car_model.getZ()
                line_seg.setThickness(2)
                line_color = [1.0, 0.0, 0.0]
                out_offset = 0.02
                w = self.WIDTH / 2 + out_offset
                l = self.LENGTH / 2 + out_offset
                h = self.HEIGHT / 2 + out_offset
                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(-w, l, h + zoffset)
                line_seg.drawTo(-w, l, -h + zoffset)
                line_seg.drawTo(w, l, -h + zoffset)
                line_seg.drawTo(w, l, h + zoffset)
                line_seg.drawTo(-w, l, -h + zoffset)
                line_seg.moveTo(-w, l, h + zoffset)
                line_seg.drawTo(w, l, -h + zoffset)

                line_seg.moveTo(w, -l, h + zoffset)
                line_seg.drawTo(-w, -l, h + zoffset)
                line_seg.drawTo(-w, -l, -h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_seg.drawTo(w, -l, h + zoffset)
                line_seg.moveTo(-w, -l, 0 + zoffset)
                line_seg.drawTo(w, -l, 0 + zoffset)
                line_seg.moveTo(0, -l, h + zoffset)
                line_seg.drawTo(0, -l, -h + zoffset)

                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(w, -l, h + zoffset)
                line_seg.moveTo(-w, l, h + zoffset)
                line_seg.drawTo(-w, -l, h + zoffset)
                line_seg.moveTo(-w, l, -h + zoffset)
                line_seg.drawTo(-w, -l, -h + zoffset)
                line_seg.moveTo(w, l, -h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_np = NodePath(line_seg.create(True))
                line_material = Material()
                line_material.setBaseColor(LVecBase4(*line_color[:3], 1))
                line_np.setMaterial(line_material, True)
                line_np.reparentTo(self.origin)

            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                     self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                     self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.))
                material.setMetallic(self.MATERIAL_METAL_COEFF)
                material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
                material.setRefractiveIndex(1.5)
                material.setRoughness(self.MATERIAL_ROUGHNESS)
                material.setShininess(self.MATERIAL_SHININESS)
                material.setTwoside(False)
                self.origin.setMaterial(material, True)

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        self._node_path_list.append(wheel_np)

        # PZH: Skip the wheel model
        # if self.render:
        #     model = 'right_tire_front.gltf' if front else 'right_tire_back.gltf'
        #     model_path = AssetLoader.file_path("models", os.path.dirname(self.path[0]), model)
        #     wheel_model = self.loader.loadModel(model_path)
        #     wheel_model.setTwoSided(self.TIRE_TWO_SIDED)
        #     wheel_model.reparentTo(wheel_np)
        #     wheel_model.set_scale(1 * self.TIRE_MODEL_CORRECT if left else -1 * self.TIRE_MODEL_CORRECT)
        wheel = self.system.createWheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))

        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(self.SUSPENSION_LENGTH)
        wheel.setSuspensionStiffness(self.SUSPENSION_STIFFNESS)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel_friction = self.config[
            "wheel_friction"] if not self.config["no_wheel_friction"] else 0
        wheel.setFrictionSlip(wheel_friction)
        wheel.setRollInfluence(0.5)
        return wheel


def random_vehicle_type(np_random, p=None):
    v_type = {
        "s": SVehicle,
        "m": MVehicle,
        "l": LVehicle,
        "xl": XLVehicle,
        "default": DefaultVehicle,
    }
    if p:
        assert len(p) == len(v_type), \
            "This function only allows to choose a vehicle from 6 types: {}".format(v_type.keys())
    prob = [1 / len(v_type) for _ in range(len(v_type))] if p is None else p
    return v_type[np_random.choice(list(v_type.keys()), p=prob)]


vehicle_type = {
    "s": SVehicle,
    "m": MVehicle,
    "l": LVehicle,
    "xl": XLVehicle,
    "default": DefaultVehicle,
    "history_default": HistoryDefaultVehicle,
    "bicycle_default": KinematicBicycleVehicle,
    "static_default": StaticDefaultVehicle,
    "varying_dynamics": VaryingDynamicsVehicle,
    "varying_dynamics_bounding_box": VaryingDynamicsBoundingBoxVehicle,
    "traffic_default": TrafficDefaultVehicle
}

vehicle_class_to_type = inv_map = {v: k for k, v in vehicle_type.items()}
