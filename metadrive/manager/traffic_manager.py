import copy
import logging
from collections import namedtuple
from typing import Dict, List, Optional

import math
import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")

class HistoryBuffer:
    """
    시간 기반 버퍼를 관리하여 최근 time_duration만큼의 이웃 차량 정보를 저장한다.
    각 스텝마다 update()를 호출하여 새로운 데이터를 추가하고,
    오래된 데이터는 time_duration을 초과하면 제거한다.

    Attributes:
        time_duration (float): 저장할 총 시간 (초 단위).
        time_gap_per_step (float): 한 스텝이 시뮬레이션 시간에서 몇 초에 해당하는지 (예: 0.1초).
        output_time_gap (float): 버퍼에서 최종 조회 시 샘플링할 시간 간격 (예: 0.1초).
        neighbor_agents_past_all (List[np.ndarray]): 각 시점의 이웃 차량 배열을 저장.
            길이는 계속 증가하지만 time_duration을 넘어서는 오래된 부분은 제거함.
            각 원소는 shape (N, 8)인 numpy array. (id, v_x, v_y, heading, width, length, x, y)
        neighbor_agents_types_all (List[List[str]]): 이웃 차량의 타입 목록(예: 차량 클래스 이름 등).
            neighbor_agents_past_all와 같은 길이를 가짐.
    """

    def __init__(self, time_duration: float, time_gap_per_step: float,
                 output_time_gap: float) -> None:
        """
        초기화.

        Args:
            time_duration (float): 버퍼에 저장할 총 시간 (초).
            time_gap_per_step (float): 한 스텝이 시뮬레이션 시간에서 몇 초인지 (예: 0.1).
            output_time_gap (float): 최종 데이터를 얻을 때 샘플링할 간격.
        Raises:
            AssertionError: time_duration이 output_time_gap으로 나누어떨어지지 않거나,
                time_gap_per_step이 output_time_gap으로 나누어떨어지지 않을 경우.
        """
        assert abs(time_duration / output_time_gap - round(time_duration / output_time_gap)) < 1e-6, \
            "time_duration must be divisible by output_time_gap"
        assert abs(time_gap_per_step / output_time_gap - round(time_gap_per_step / output_time_gap)) < 1e-6, \
            "time_gap_per_step must be divisible by output_time_gap"
        self.time_duration = time_duration
        self.time_gap_per_step = time_gap_per_step
        self.output_time_gap = output_time_gap

        # 버퍼
        self.neighbor_agents_past_all: List[np.ndarray] = []
        self.neighbor_agents_types_all: List[List[str]] = []

        # time_elapsed를 추적 (after_step마다 time_gap_per_step만큼 증가한다고 가정)
        self._time_elapsed_list: List[float] = []  # 각 step별 누적시간

    def reset(self) -> None:
        """
        버퍼 초기화.
        """
        self.neighbor_agents_past_all.clear()
        self.neighbor_agents_types_all.clear()
        self._time_elapsed_list.clear()

    def update(self, neighbor_agents: np.ndarray,
               neighbor_types: List[str]) -> None:
        """
        버퍼에 새로운 이웃 차량 정보를 추가하고, time_duration을 초과한 오래된 데이터를 제거한다.

        Args:
            neighbor_agents (np.ndarray): shape (N, 8)인 numpy array (id, vx, vy, heading, width, length, x, y)
            neighbor_types (List[str]): 길이 N인 리스트. 각 에이전트 타입(차량 클래스명 등)
        """
        # 시뮬레이션 시간이 누적되어 들어온다고 가정
        current_time = 0.0 if len(
            self._time_elapsed_list) == 0 else (self._time_elapsed_list[-1] +
                                                self.time_gap_per_step)

        self.neighbor_agents_past_all.append(neighbor_agents)
        self.neighbor_agents_types_all.append(neighbor_types)
        self._time_elapsed_list.append(current_time)

        # 오래된 데이터 제거
        # time_elapsed_list 맨 끝 - 맨 앞 <= time_duration이 되도록 남긴다
        while len(self._time_elapsed_list) > 0 and \
                (self._time_elapsed_list[-1] - self._time_elapsed_list[0]) > self.time_duration:
            self.neighbor_agents_past_all.pop(0)
            self.neighbor_agents_types_all.pop(0)
            self._time_elapsed_list.pop(0)

    def get_neighbor_agents_past(self) -> List[np.ndarray]:
        """
        버퍼에서 output_time_gap 간격으로 추출한 이웃 차량 데이터 목록을 반환한다.
        예: time_gap_per_step=0.1, output_time_gap=0.1 → 모든 스텝 반환
            time_gap_per_step=0.05, output_time_gap=0.1 → 두 스텝에 한 번씩 반환

        Returns:
            List[np.ndarray]: 필터링된 이웃 차량 목록 (time 순서). shape (N, 8).
        """
        if len(self._time_elapsed_list) == 0:
            return []

        # output_time_gap을 기준으로 샘플링
        result: List[np.ndarray] = []
        sampling_interval_steps = int(
            round(self.output_time_gap / self.time_gap_per_step))
        leftover_ = (len(self.neighbor_agents_past_all) -1) % sampling_interval_steps
        for i in range(leftover_, len(self.neighbor_agents_past_all),
                       sampling_interval_steps):
            result.append(self.neighbor_agents_past_all[i])
        return result

    def get_neighbor_agents_types(self) -> List[List[str]]:
        """
        버퍼에서 output_time_gap 간격으로 추출한 이웃 차량 타입 목록.

        Returns:
            List[List[str]]: 이웃 차량 타입 (time 순서).
        """
        if len(self._time_elapsed_list) == 0:
            return []

        result: List[List[str]] = []
        sampling_interval_steps = int(
            round(self.output_time_gap / self.time_gap_per_step))
        leftover_ = (len(self.neighbor_agents_past_all) - 1) % sampling_interval_steps
        for i in range(leftover_, len(self.neighbor_agents_types_all),
                       sampling_interval_steps):
            result.append(self.neighbor_agents_types_all[i])
        return result

class TrafficMode:
    # Traffic vehicles will be spawned once
    Basic = "basic"

    # Traffic vehicles will be respawned, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once, and will be triggered when agent comes close
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"


class PGTrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(PGTrafficManager, self).__init__()
        self._id_map = {}
        self._next_id = 1
        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None

        # 예시: time_duration=2., time_gap_per_step=0.1, output_time_gap=0.1
        self.history_buffer = HistoryBuffer(time_duration=2.0,
                                            time_gap_per_step=0.1,
                                            output_time_gap=0.1)
        # --- 새로 생성된 차량에 대해 버퍼 업데이트 ---
        neighbor_agents, neighbor_types = self._collect_neighbor_data()
        self.history_buffer.update(neighbor_agents, neighbor_types)

    def _get_float_id_for_vehicle(self, v_id_str: str) -> float:
        if v_id_str not in self._id_map:
            self._id_map[v_id_str] = self._next_id
            self._next_id += 1
        return float(self._id_map[v_id_str])

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        # TODO: reset 해도 id를 유지해야 하는 부분이 있는지 궁금
        self._id_map.clear()
        self._next_id = 1
        self.history_buffer.reset()
        map = self.current_map
        logging.debug("load scene {}".format(
            "Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)

        logging.debug(
            f"Resetting Traffic Manager with mode {self.mode} and density {traffic_density}"
        )
        if self.mode in {TrafficMode.Basic, TrafficMode.Respawn}:
            self._create_basic_vehicles(map, traffic_density)
        elif self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            self._create_trigger_vehicles(map, traffic_density)
        else:
            raise ValueError(f"No such mode named {self.mode}")
        # --- 새로 생성된 차량에 대해 버퍼 업데이트 ---
        neighbor_agents, neighbor_types = self._collect_neighbor_data()
        self.history_buffer.update(neighbor_agents, neighbor_types)

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v in engine.agent_manager.active_agents.values():
                if len(self.block_triggered_vehicles) > 0:
                    ego_lane_idx = v.lane_index[:-1]
                    ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                    if ego_road == self.block_triggered_vehicles[
                            -1].trigger_road:
                        block_vehicles = self.block_triggered_vehicles.pop()
                        self._traffic_vehicles += list(
                            self.get_objects(block_vehicles.vehicles).values())

        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                v_to_remove.append(v)

        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)

            # Spawn new vehicles to replace the removed one
            if self.mode in {TrafficMode.Respawn, TrafficMode.Hybrid}:
                lane = self.respawn_lanes[self.np_random.randint(
                    0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {
                    "spawn_lane_index": lane_idx,
                    "spawn_longitude": long
                }
                new_v = self.spawn_object(vehicle_type,
                                          vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(new_v.id, IDMPolicy, new_v,
                                self.generate_seed())
                self._traffic_vehicles.append(new_v)
        neighbor_agents, neighbor_types = self._collect_neighbor_data()
        self.history_buffer.update(neighbor_agents, neighbor_types)

        return dict()

    def _collect_neighbor_data(self) -> (np.ndarray, List[str]):
        """
        이웃 차량 정보를 수집. 여기서는 간단히 self._traffic_vehicles 중,
        shape (N,8)를 구성해본다.

        Returns:
            np.ndarray: shape (N,8), columns: [id, vx, vy, heading, width, length, x, y]
            List[str]: 차량의 타입 문자열. e.g. vehicle_type.__name__ 등
        """
        if len(self._traffic_vehicles) == 0:
            return np.zeros((0, 8), dtype=np.float32), []

        data_list = []
        type_list = []
        for v in self._traffic_vehicles:
            # id를 임의로 hash했거나 int로 변환해야 한다. 여기서는 그냥 id를 float로 cast 불가능하면 0
            # toy example
            # velocity -> v.velocity
            # heading -> v.heading_theta
            # width -> v.WIDTH
            # length -> v.LENGTH
            # x -> v.position[0]
            # y -> v.position[1]
            vehicle_id = self._get_float_id_for_vehicle(v.id)


            vx = v.velocity[0]  # m/s
            vy = v.velocity[1]  # m/s
            heading = v.heading_theta  # rad
            w = v.WIDTH
            l = v.LENGTH
            x_ = v.position[0]
            y_ = v.position[1]
            row = [vehicle_id, vx, vy, heading, w, l, x_, y_]
            data_list.append(row)
            type_list.append(type(v).__name__)

        arr = np.array(data_list, dtype=np.float32)
        return arr, type_list

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(PGTrafficManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self.block_triggered_vehicles = []
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        if self.mode in {TrafficMode.Basic, TrafficMode.Respawn}:
            return len(self._traffic_vehicles)
        return sum(
            len(block_vehicle_set.vehicles)
            for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        # collect other vehicles
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    traffic_states[vehicle.index] = vehicle.get_state()
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {
            k: v.get_state() for k, v in active_obj.items()
        }
        states[TARGET_VEHICLES] = merge_dicts(states[TARGET_VEHICLES], {
            k: v.get_state() for k, v in pending_obj.items()
        },
                                              allow_new_keys=True)
        states[TARGET_VEHICLES] = merge_dicts(states[TARGET_VEHICLES], {
            k: v_count[0].get_state() for k, v_count in dying_obj.items()
        },
                                              allow_new_keys=True)

        states[OBJECT_TO_AGENT] = copy.deepcopy(
            self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(
            self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def _propose_vehicle_configs(self, lane: AbstractLane):
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {
                "spawn_lane_index": lane.index,
                "spawn_longitude": long,
                "enable_reverse": False
            }
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    def _create_basic_vehicles(self, map: BaseMap, traffic_density: float):
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            for long in vehicle_longs[:int(
                    np.ceil(traffic_density * len(vehicle_longs)))]:
                # if self.np_random.rand() > traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                #     # Do special handling for ramp, and there must be vehicles created there
                #     continue
                vehicle_type = self.random_vehicle_type()
                traffic_v_config = {
                    "spawn_lane_index": lane.index,
                    "spawn_longitude": long
                }
                traffic_v_config.update(
                    self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type,
                                             vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(random_v.id, IDMPolicy, random_v,
                                self.generate_seed())
                self._traffic_vehicles.append(random_v)

    def _create_trigger_vehicles(self, map: BaseMap,
                                 traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config[
                    "need_inverse_traffic"] and block.ID in [
                        "S", "C", "r", "R"
                    ]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(
                            self.engine, "object_manager"
                    ) and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(
                        l)

            # How many vehicles should we spawn in this block?
            total_length = sum(
                [lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length /
                                                self.VEHICLE_GAP))
            total_vehicles = int(
                math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(
                total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(
                    self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type,
                                             vehicle_config=v_config)
                seed = self.generate_seed()
                self.add_policy(random_v.id, IDMPolicy, random_v, seed)
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road,
                                           vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def _get_available_respawn_lanes(self, map: BaseMap) -> list:
        """
        Used to find some respawn lanes
        :param map: select spawn lanes from this map
        :return: respawn_lanes
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads:
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random,
                                           [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self._traffic_vehicles.__repr__()

    @property
    def vehicles(self):
        return list(
            self.engine.get_objects(
                filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        if not self.random_traffic:
            super(PGTrafficManager, self).seed(random_seed)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def get_state(self):
        ret = super(PGTrafficManager, self).get_state()
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        flat = []
        for b_v in self.block_triggered_vehicles:
            flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node,
                         b_v.vehicles))
        ret["block_triggered_vehicles"] = flat
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        super(PGTrafficManager, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([
                old_name_to_current[name] for name in state["_traffic_vehicles"]
            ]).values())
        self.block_triggered_vehicles = [
            BlockVehicles(trigger_road=Road(s, e),
                          vehicles=[old_name_to_current[name]
                                    for name in v])
            for s, e, v in state["block_triggered_vehicles"]
        ]


# For compatibility check
TrafficManager = PGTrafficManager


class MixedPGTrafficManager(PGTrafficManager):

    def _create_basic_vehicles(self, *args, **kwargs):
        raise NotImplementedError()

    def _create_trigger_vehicles(self, map: BaseMap,
                                 traffic_density: float) -> None:
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config[
                    "need_inverse_traffic"] and block.ID in [
                        "S", "C", "r", "R"
                    ]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(
                            self.engine, "object_manager"
                    ) and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(
                        l)

            # How many vehicles should we spawn in this block?
            total_length = sum(
                [lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length /
                                                self.VEHICLE_GAP))
            total_vehicles = int(
                math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(
                total_vehicles, len(potential_vehicle_configs))]

            from metadrive.policy.idm_policy import IDMPolicy
            from metadrive.policy.expert_policy import ExpertPolicy
            # print("===== We are initializing {} vehicles =====".format(len(selected)))
            # print("Current seed: ", self.engine.global_random_seed)
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(
                    self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type,
                                             vehicle_config=v_config)
                if self.np_random.random(
                ) < self.engine.global_config["rl_agent_ratio"]:
                    # print("Vehicle {} is assigned with RL policy!".format(random_v.id))
                    self.add_policy(random_v.id, ExpertPolicy, random_v,
                                    self.generate_seed())
                else:
                    self.add_policy(random_v.id, IDMPolicy, random_v,
                                    self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road,
                                           vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()
