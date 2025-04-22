from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network import Road
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metadrive.engine.engine_utils import get_engine
from metadrive.manager.base_manager import BaseManager
import numpy as np


class TrafficObjectManager(BaseManager):
    """
    This class is used to manager all static object, such as traffic cones, warning tripod.
    """
    PRIORITY = 9

    # the distance between break-down vehicle and alert
    ALERT_DIST = 10

    # accident scene setting
    ACCIDENT_AREA_LEN = 10

    # distance between two cones
    CONE_LONGITUDE = 2
    CONE_LATERAL = 1
    PROHIBIT_SCENE_PROB = 0.67  # the reset is the probability of break_down_scene

    def __init__(self):
        super(TrafficObjectManager, self).__init__()
        self.accident_prob = 0.
        self.accident_lanes = []

    def before_reset(self):
        """
        Clear all objects in th scene
        """
        super(TrafficObjectManager, self).before_reset()
        self.accident_prob = self.engine.global_config["accident_prob"]

    def reset(self):
        """
        Generate an accident scene or construction scene on block
        :return: None
        """
        self.accident_lanes = []
        engine = get_engine()
        accident_prob = self.accident_prob
        if abs(accident_prob - 0.0) < 1e-2:
            return
        for block in engine.current_map.blocks:
            if type(block) not in [
                    Straight, Curve, InRampOnStraight, OutRampOnStraight
            ]:
                # blocks with exists do not generate accident scene
                continue
            if self.np_random.rand() > accident_prob:
                # prob filter
                continue

            road_1 = Road(block.pre_block_socket.positive_road.end_node,
                          block.road_node(0, 0))
            road_2 = Road(block.road_node(0, 0), block.road_node(
                0, 1)) if not isinstance(block, Straight) else None

            if self.np_random.rand() > self.PROHIBIT_SCENE_PROB:
                accident_road = self.np_random.choice([
                    road_1, road_2
                ]) if not isinstance(block, Curve) else road_2
                accident_road = road_1 if accident_road is None else accident_road
                is_ramp = isinstance(block, InRampOnStraight) or isinstance(
                    block, OutRampOnStraight)
                on_left = True if self.np_random.rand() > 0.5 or (
                    accident_road is road_2 and is_ramp) else False
                accident_lane_idx = 0 if on_left else -1
                lane = accident_road.get_lanes(
                    engine.current_map.road_network)[accident_lane_idx]
                longitude = lane.length - self.ACCIDENT_AREA_LEN - 5

                lateral_len = engine.current_map.config[
                    engine.current_map.LANE_WIDTH]

                lane = engine.current_map.road_network.get_lane(
                    accident_road.lane_index(accident_lane_idx))
                self.accident_lanes.append(
                    accident_road.get_lanes(
                        engine.current_map.road_network)[accident_lane_idx])
                self.prohibit_scene(lane, longitude, lateral_len, on_left)
            else:
                accident_road = self.np_random.choice([road_1, road_2])
                accident_road = road_1 if accident_road is None else accident_road
                is_ramp = isinstance(block, InRampOnStraight) or isinstance(
                    block, OutRampOnStraight)
                on_left = True if self.np_random.rand() > 0.5 or (
                    accident_road is road_2 and is_ramp) else False
                lanes = accident_road.get_lanes(engine.current_map.road_network)
                if len(lanes) - 1 == 0:
                    accident_lane_idx = -1
                else:
                    accident_lane_idx = self.np_random.randint(
                        0,
                        len(lanes) - 1) if on_left else -1
                lane = lanes[accident_lane_idx]
                self.accident_lanes.append(
                    accident_road.get_lanes(
                        engine.current_map.road_network)[accident_lane_idx])
                longitude = self.np_random.rand(
                ) * lane.length / 2 + lane.length / 2
                if self.np_random.rand() > 0.5:
                    self.break_down_scene(lane, longitude)
                else:
                    self.barrier_scene(lane, longitude)

    def break_down_scene(self, lane: AbstractLane, longitude: float):
        v_config = {
            "spawn_lane_index": lane.index,
            "spawn_longitude": float(longitude)
        }
        breakdown_vehicle = self.spawn_object(
            self.engine.traffic_manager.random_vehicle_type(),
            vehicle_config=v_config)
        breakdown_vehicle.set_break_down()
        longitude = longitude - self.ALERT_DIST
        lateral = 0
        self.spawn_object(
            TrafficWarning,
            lane=lane,
            position=lane.position(longitude, lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(longitude))

    def barrier_scene(self, lane, longitude):
        longitude = longitude
        lateral = 0
        self.spawn_object(
            TrafficBarrier,
            lane=lane,
            position=lane.position(longitude, lateral),
            static=self.engine.global_config["static_traffic_object"],
            heading_theta=lane.heading_theta_at(longitude))

    def prohibit_scene(self,
                       lane: AbstractLane,
                       longitude_position: float,
                       lateral_len: float,
                       on_left=False):
        """
        Generate an accident scene on the most left or most right lane
        :param lane object
        :param longitude_position: longitude position of the accident on the lane
        :param lateral_len: the distance that traffic cones extend on lateral direction
        :param on_left: on left or right side
        :return: None
        """
        lat_num = int(lateral_len / self.CONE_LATERAL)
        longitude_num = int(self.ACCIDENT_AREA_LEN / self.CONE_LONGITUDE)
        lat_1 = [lat * self.CONE_LATERAL for lat in range(lat_num)]
        lat_2 = [lat_num * self.CONE_LATERAL] * (longitude_num + 1)
        lat_3 = [(lat_num - lat - 1) * self.CONE_LATERAL
                 for lat in range(int(lat_num))]

        total_long_num = lat_num * 2 + longitude_num + 1
        pos = [
            (long * self.CONE_LONGITUDE, lat - lane.width / 2)
            for long, lat in zip(
                range(-int(total_long_num / 2), int(total_long_num /
                                                    2)), lat_1 + lat_2 + lat_3)
        ]
        left = 1 if on_left else -1
        for p in pos:
            p_ = (p[0] + longitude_position, left * p[1])
            position = lane.position(p_[0], p_[1])
            heading_theta = lane.heading_theta_at(p_[0])
            self.spawn_object(
                TrafficCone,
                lane=lane,
                position=position,
                heading_theta=heading_theta,
                static=self.engine.global_config["static_traffic_object"])

    def set_state(self, state: dict, old_name_to_current=None):
        """
        Copied from super(). Restoring some states before reassigning value to spawned_objets
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        if old_name_to_current is None:
            old_name_to_current = {key: key for key in state.keys()}
        spawned_objects = state["spawned_objects"]
        ret = {}
        for name, class_name in spawned_objects.items():
            current_name = old_name_to_current[name]
            name_obj = self.engine.get_objects([current_name])
            assert current_name in name_obj and name_obj[
                current_name].class_name == class_name, "Can not restore mappings!"
            # Restore some internal states
            name_obj[
                current_name].lane = self.engine.current_map.road_network.get_lane(
                    name_obj[current_name].lane.index)

            ret[current_name] = name_obj[current_name]
        self.spawned_objects = ret

    def get_static_object_array(self, ego_vehicle, max_obj = 5) -> np.ndarray:
        """
        최대 5개의 정적오브젝트를, 에고 로컬좌표계로 변환한 뒤 반환.
        반환 shape=(5,10). 각 행은 [ x_local, y_local, cos(Δheading), sin(Δheading),
                                 width, length, one_hot(4차원) ]
        한편, x_local,y_local 및 heading은 "에고차" 기준으로 변환됨.
        """
        # 1) 현재 매니저가 가진 모든 정적 object
        all_objs = list(self.spawned_objects.values())

        # 2) 필요 객체 필터링 + 속성 추출: (x,y,heading,width,length,type_index)
        static_candidates = []
        for obj in all_objs:
            if isinstance(obj, TrafficWarning):
                obj_type = 0
            elif isinstance(obj, TrafficBarrier):
                obj_type = 1
            elif isinstance(obj, TrafficCone):
                obj_type = 2
            else:
                obj_type = 3  # generic

            # object position, heading, width, length 추출
            x, y = obj.position
            heading = obj.heading_theta
            w = getattr(obj, "WIDTH", 1.0)
            l = getattr(obj, "LENGTH", 1.0)
            static_candidates.append((x, y, heading, w, l, obj_type))

        if len(static_candidates) == 0:
            return np.zeros((max_obj, 10), dtype=np.float32)

        # 배열화 => shape=(N,6)
        static_arr = np.array(static_candidates, dtype=np.float32)
        # columns: 0=x,1=y,2=heading,3=width,4=length,5=type_idx

        # 3) 에고 위치, heading
        ego_x, ego_y = ego_vehicle.rear_axle_xy
        ego_yaw = ego_vehicle.heading_theta  # 라디안

        # 4) 에고와의 거리 계산
        dx = static_arr[:, 0] - ego_x  # (N,)
        dy = static_arr[:, 1] - ego_y  # (N,)
        dist_arr = np.hypot(dx, dy)  # (N,)
        idx_sorted = np.argsort(dist_arr)  # 거리 오름차순 정렬 idx


        sel_len = min(max_obj, len(idx_sorted))
        selected_idx = idx_sorted[:sel_len]

        # 최종 결과 (5,10)
        result = np.zeros((max_obj, 10), dtype=np.float32)

        # 5) 이제 local 변환
        # ego좌표계 (dx,dy) => local_x, local_y
        # heading relative => heading - ego_yaw
        # rotation (batch)
        # local_x = dx*cos(ego_yaw) + dy*sin(ego_yaw)
        # local_y = -dx*sin(ego_yaw) + dy*cos(ego_yaw)
        dx_sel = dx[selected_idx]
        dy_sel = dy[selected_idx]
        c = np.cos(ego_yaw)
        s = np.sin(ego_yaw)
        local_x = dx_sel * c + dy_sel * s
        local_y = -dx_sel * s + dy_sel * c

        heading_vals = static_arr[selected_idx, 2]  # (sel_len,) object heading
        rel_heading = heading_vals - ego_yaw  # relative heading
        cos_local = np.cos(rel_heading)
        sin_local = np.sin(rel_heading)

        width_vals = static_arr[selected_idx, 3]
        length_vals = static_arr[selected_idx, 4]

        type_int = static_arr[selected_idx, 5].astype(np.int64)  # (sel_len,)
        one_hot_mat = np.eye(4, dtype=np.float32)[type_int]  # (sel_len,4)

        # 6) 채우기
        result[:sel_len, 0] = local_x
        result[:sel_len, 1] = local_y
        result[:sel_len, 2] = cos_local
        result[:sel_len, 3] = sin_local
        result[:sel_len, 4] = width_vals
        result[:sel_len, 5] = length_vals
        result[:sel_len, 6:10] = one_hot_mat

        return result
