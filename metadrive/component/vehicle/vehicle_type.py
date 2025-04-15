import platform

from panda3d.core import LineSegs, NodePath
from panda3d.core import Material, Vec3, LVecBase4

from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.constants import Semantics
from metadrive.engine.asset_loader import AssetLoader

import numpy as np
import math

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.math import clip, wrap_to_pi
from metadrive.engine.logger import get_logger

logger = get_logger()



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




class KinematicBicycleVehicle(BaseVehicle):
    """
    이 차량 클래스는, '가속도 + 조향속도'로 ego 차량 상태를 업데이트하는
    운동학적 자전거 모델(kineamtic bicycle)을 구현한 예시입니다.
    """

    def __init__(self, vehicle_config=None, name=None, random_seed=None,
                 position=None, heading=None, _calling_reset=True):
        super().__init__(vehicle_config=vehicle_config,
                         name=name,
                         random_seed=random_seed,
                         position=position,
                         heading=heading,
                         _calling_reset=_calling_reset)
        # 조향각 (rad)
        self.steering_angle = 0.0
        # 선속도 (m/s)
        self.current_speed = 0.0
        # 바퀴 간 거리 (휠베이스). 예: 보통은 vehicle 길이와 유사하거나 작은 값
        # 여기서는 일단 self.LENGTH 를 사용하거나, 필요하면 config 에서 wheelbase 를 읽을 수도 있음
        self.wheelbase = self.LENGTH

        # 만약 Bullet 물리에 의한 이동을 완전히 끄고 싶다면,
        # reset() 시점에 self.set_static(True) 등을 적용해서 '정적 바디'로 두는 방법을 추천합니다.

    def before_step(self, action=None):
        """
        매 스텝에, 액션 = [가속도(m/s^2), 조향속도(rad/s)]를 받아
        kinematic bicycle 식으로 상태를 적분한 뒤,
        self.set_position() / self.set_heading_theta() / self.set_velocity() / self.set_angular_velocity() 등으로
        위치, 헤딩, 속도, 각속도를 반영합니다.
        """
        # 1) bullet 의 기본 로직처럼 action을 -1~1로 클립 후, 실제 accel/steer_rate로 변환해도 되지만,
        #    여기서는 "이미 [accel, steering_rate]" 로 들어온다고 가정
        if action is None:
            # 아무 입력이 없으면 가속도=0, 조향속도=0 으로 처리
            action = [0.0, 0.0]

        accel = float(action[0])           # m/s^2
        steering_rate = float(action[1])   # rad/s

        # 2) 시뮬레이션에서 dt를 가져옴: decision_repeat × physics_world_step_size
        engine_config = self.engine.global_config
        decision_repeat = engine_config["decision_repeat"]
        physics_step = engine_config["physics_world_step_size"]
        dt = decision_repeat * physics_step

        # 3) 현재 스티어링 각도(steering_angle)에 steering_rate * dt 더함
        self.steering_angle += steering_rate * dt

        # 스티어링 최대각 (기존 self.max_steering 는 deg 단위이므로 rad 로 바꿔야 함)
        max_steer_rad = float(self.max_steering) * math.pi / 180.0
        self.steering_angle = clip(self.steering_angle, -max_steer_rad, max_steer_rad)

        # 4) 현재 헤딩/속도 가져오기
        old_heading = self.heading_theta  # rad
        old_speed = self.current_speed    # m/s

        # 5) 운동학 자전거 모델 공식: 요레이트 = v * tan(delta) / wheelbase
        heading_rate = old_speed * math.tan(self.steering_angle) / self.wheelbase

        # 6) 속도 적분: new_speed = old_speed + accel * dt
        new_speed = old_speed + accel * dt
        # 속도 하한(0) 제한 (필요시 clip)
        if new_speed < 0.0:
            new_speed = 0.0

        # 7) 헤딩 적분: new_heading = old_heading + heading_rate * dt
        new_heading = old_heading + heading_rate * dt
        new_heading = wrap_to_pi(new_heading)

        # 8) 위치 업데이트 (전방속도 * dt, heading 기준)
        #    여기서는 old_heading 기준(또는 new_heading 근사) 사용가능
        #    취향에 따라 midpoint 방정식 등도 가능
        old_x, old_y = self.position
        dx = old_speed * math.cos(old_heading) * dt
        dy = old_speed * math.sin(old_heading) * dt
        new_x = old_x + dx
        new_y = old_y + dy

        # 9) 차량 상태에 반영
        # (A) 위치/헤딩
        self.set_position((new_x, new_y))
        self.set_heading_theta(new_heading, in_rad=True)

        # (B) 선속도/조향각 업데이트
        self.current_speed = new_speed  # 내부적으로 보관하는 현재 속도(m/s)

        # (C) Bullet 엔진에 속도/각속도 동기화 (렌더/센서 등)
        #     "정적 바디"일 경우 Bullet 이 실제로 힘을 가하지 않으므로,
        #     충돌 감지 용도로만 velocity가 필요하다면 아래처럼 set
        heading_vec = np.array([math.cos(new_heading), math.sin(new_heading)])
        self.set_velocity(heading_vec, value=new_speed)

        # 각속도 (yaw rate). Bullet 의 setAngularVelocity 는 [0,0,angular_vel_z]
        self.set_angular_velocity(heading_rate, in_rad=True)

        # 10) 만약 "조향각"도 Bullet Vehicle 의 setSteeringValue 에 반영해주면,
        #     시각적으로 바퀴가 돌아가는 모습(차모델 휠)에 반영 가능 (필요시):
        #     --> self.system.setSteeringValue(self.steering_angle, 0)
        #         self.system.setSteeringValue(self.steering_angle, 1)

        # parent 클래스인 BaseVehicle.before_step() 처럼, 추가로 step_info 를 dict로 반환할 수도 있음
        step_info = {
            "raw_action": (accel, steering_rate),
            "steering_angle_rad": self.steering_angle,
            "speed_m_s": self.current_speed,
            "heading_rate_rad_s": heading_rate,
        }
        return step_info

    def reset(self, vehicle_config=None, name=None, random_seed=None,
              position=None, heading=0.0, *args, **kwargs):
        # super().reset(...) 로 기본적인 vehicle 초기화
        super().reset(vehicle_config=vehicle_config,
                      name=name,
                      random_seed=random_seed,
                      position=position,
                      heading=heading,
                      *args,
                      **kwargs)
        # 추가로, 조향각과 속도를 0으로 초기화
        self.steering_angle = 0.0
        self.current_speed = 0.0

        # 만약 Bullet 물리를 완전히 끄고 싶다면, 여기서 set_static(True) 호출
        self.set_static(True)

        # 예: wheel friction, engine force 등도 무의미하게 만듦
        self.config["no_wheel_friction"] = True
        #    --> 필요 시, config에서 세팅

        # 필요한 경우 상위 reset()이 반환하는 값(step_info 등) 반환
        # 그러나 보통 base_vehicle.reset()은 None 리턴,
        # 굳이 반환하고 싶다면 return super(...).reset(...)



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
    "bicycle_default": KinematicBicycleVehicle,
    "static_default": StaticDefaultVehicle,
    "varying_dynamics": VaryingDynamicsVehicle,
    "varying_dynamics_bounding_box": VaryingDynamicsBoundingBoxVehicle,
    "traffic_default": TrafficDefaultVehicle
}

vehicle_class_to_type = inv_map = {v: k for k, v in vehicle_type.items()}
