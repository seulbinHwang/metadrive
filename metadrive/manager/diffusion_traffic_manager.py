import copy
import logging
from collections import namedtuple
from typing import Dict, List, Optional
from panda3d.core import LVector3
import math

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative, TrackedObjectType, AgentInternalIndex, EgoInternalIndex
import numpy as np
from typing import List

from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.lqr_policy import LQRPolicy  # ← 이미 구현돼 있다고 가정
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import RENDER_MODE_NONE

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


def _filter_agents_array(agents, reverse: bool = False):
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param agents: The past agents in the scene. A list of [num_frames] arrays, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """
    target_array = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):

        rows = []
        for j in range(agents[i].shape[0]):
            if target_array.shape[0] > 0:
                agent_id: float = float(
                    agents[i][j, int(AgentInternalIndex.track_token())])
                is_in_target_frame: bool = bool(
                    (agent_id == target_array[:,
                                              AgentInternalIndex.track_token()]
                    ).max())
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        if len(rows) > 0:
            agents[i] = np.stack(rows)
        else:
            agents[i] = np.empty((0, agents[i].shape[1]), dtype=np.float32)

    return agents


def _pad_agent_states(agent_trajectories, reverse: bool):
    """
    Pads the agent states with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """

    track_id_idx = 0
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    key_frame = agent_trajectories[0]

    id_row_mapping: Dict[int, int] = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val)] = idx

    current_state = np.zeros((key_frame.shape[0], key_frame.shape[1]),
                             dtype=np.float64)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]

        # Update current frame
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx])]
            current_state[mapped_row, :] = frame[row_idx, :]

        # Save current state
        agent_trajectories[idx] = current_state.copy()

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    return agent_trajectories


class HistoryBuffer:
    """
    시간 기반 버퍼를 관리하여 최근 time_duration만큼의 이웃 차량 정보를 저장한다.
    각 스텝마다 update()를 호출하여 새로운 데이터를 추가하고,
    오래된 데이터는 time_duration을 초과하면 제거한다.

    Attributes:
        time_duration (float): 저장할 총 시간 (초 단위).
        time_gap_per_step (float): 한 스텝이 시뮬레이션 시간에서 몇 초에 해당하는지 (예: 0.1초).
        output_time_gap (float): 버퍼에서 최종 조회 시 샘플링할 시간 간격 (예: 0.1초).
        max_slot (int): time_duration을 time_gap_per_step로 나눈 최대 슬롯 수 + 1
        neighbor_agents_past_all (List[np.ndarray]): 각 시점의 이웃 차량 배열(길이 max_slot).
            각 원소는 shape (N, 8)인 numpy array. (id, v_x, v_y, heading, width, length, x, y)
        neighbor_agents_types_all (List[List[TrackedObjectType]]): 이웃 차량 타입 목록(길이 max_slot).
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
        self.max_slot = int(round(
            self.time_duration / self.time_gap_per_step)) + 1

        self.neighbor_agents_past_all: List[np.ndarray] = []
        self.neighbor_agents_types_all: List[List[str]] = []
        self._time_elapsed_list: List[float] = []

        self.reset()

    def reset(self) -> None:
        """
        버퍼를 초기화하고 길이를 self.max_slot으로 맞춘다.
        아직 채워지지 않은 슬롯은 np.zeros((0,8)) / [] 로 채움.
        """
        self.neighbor_agents_past_all = [
            np.zeros((0, 8), dtype=np.float32) for _ in range(self.max_slot)
        ]
        self.neighbor_agents_types_all = [[] for _ in range(self.max_slot)]
        # time_elapsed도 동일하게 self.max_slot 크기로 초기화
        # (아직 의미 있는 시간이 아니므로 0.0으로 채우거나 None 등으로 채워도 됨)
        self._time_elapsed_list = [0.0] * self.max_slot

    def update(self, neighbor_agents: np.ndarray,
               neighbor_types: List[str]) -> None:
        """
        버퍼에 새로운 이웃 차량 정보를 추가하고, 가장 오래된 슬롯을 제거하여 길이를 유지한다.

        Args:
            neighbor_agents (np.ndarray): shape (N, 8)인 numpy array (id, vx, vy, heading, width, length, x, y)
            neighbor_types (List[str]): 길이 N인 리스트. 각 에이전트 타입(차량 클래스명 등)
        """
        # 이번 스텝 시간(누적) 계산
        current_time = 0.0 if len(self._time_elapsed_list) == 0 else \
            (self._time_elapsed_list[-1] + self.time_gap_per_step)

        # 오래된 슬롯을 pop(0)하고 새 데이터를 append
        self.neighbor_agents_past_all.pop(0)
        self.neighbor_agents_types_all.pop(0)
        self._time_elapsed_list.pop(0)

        self.neighbor_agents_past_all.append(neighbor_agents)
        self.neighbor_agents_types_all.append(neighbor_types)
        self._time_elapsed_list.append(current_time)

    def get_neighbor_agents_past(self) -> List[np.ndarray]:
        """
        버퍼에서 output_time_gap 간격으로 추출한 이웃 차량 데이터 목록을 반환한다.
        예: time_gap_per_step=0.1, output_time_gap=0.1 → 모든 스텝 반환
            time_gap_per_step=0.05, output_time_gap=0.1 → 두 스텝에 한 번씩 반환

        Returns:
            List[np.ndarray]: 필터링된 이웃 차량 목록 (time 순서). shape (N, 8).
        """
        # 길이가 0이면 빈 리스트 반환
        if len(self.neighbor_agents_past_all) == 0:
            return []

        sampling_interval_steps = int(
            round(self.output_time_gap / self.time_gap_per_step))
        # leftover 계산으로 인덱스를 맞춤
        # 마지막 프레임(가장 최근)을 반드시 포함하기 위해 leftover를 사용
        leftover_ = (len(self.neighbor_agents_past_all) -
                     1) % sampling_interval_steps

        result = []
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
        if len(self.neighbor_agents_types_all) == 0:
            return []

        sampling_interval_steps = int(
            round(self.output_time_gap / self.time_gap_per_step))
        leftover_ = (len(self.neighbor_agents_types_all) -
                     1) % sampling_interval_steps

        result = []
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

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format(
            "Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        else:
            raise ValueError("No such mode named {}".format(self.mode))

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode != TrafficMode.Respawn:
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
                if self.mode == TrafficMode.Trigger:
                    v_to_remove.append(v)
                elif self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                    v_to_remove.append(v)
                else:
                    raise ValueError("Traffic mode error: {}".format(self.mode))
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
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

        return dict()

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
        if self.mode == TrafficMode.Respawn:
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
        if self.mode != TrafficMode.Respawn:
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
        if self.mode != TrafficMode.Respawn:
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

    def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float):
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            for long in vehicle_longs[:int(
                    np.ceil(traffic_density * len(vehicle_longs)))]:
                # TODO: remove
                if len(self._traffic_vehicles) >= 10:
                    break

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

    def _create_vehicles_once(self, map: BaseMap,
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
                # TODO: remove
                if len(self._traffic_vehicles) >= 10:
                    break
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


class HistoricalBufferTrafficManager(PGTrafficManager):
    VEHICLE_GAP = 10  # m
    PRIORITY = 9

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(HistoricalBufferTrafficManager, self).__init__()
        self.str_id_to_int_id = {}
        self._next_int_id = 1

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.num_agents = 32
        self.max_ped_bike = 0
        self.time_duration = 2.0
        self.time_gap_per_step = 0.1
        self.output_time_gap = 0.1
        self.max_slot = int(round(
            self.time_duration / self.output_time_gap)) + 1
        # 예시: time_duration=2., time_gap_per_step=0.1, output_time_gap=0.1
        self.history_buffer = HistoryBuffer(
            time_duration=self.time_duration,
            time_gap_per_step=self.time_gap_per_step,
            output_time_gap=self.output_time_gap)

    def get_neighbors_history(
        self, ego_vehicle: BaseVehicle
    ) -> np.ndarray:  # shape (num_agents, num_frames, 11)
        ego_pose = np.array([
            ego_vehicle.rear_axle_xy[0], ego_vehicle.rear_axle_xy[1],
            ego_vehicle.heading_theta
        ])
        agents_states_dim = 8
        neighbor_agents_past = self.history_buffer.get_neighbor_agents_past()
        neighbor_agents_types = self.history_buffer.get_neighbor_agents_types()
        track_token_ids = self.str_id_to_int_id
        agent_history = _filter_agents_array(neighbor_agents_past, reverse=True)
        agent_types: List[TrackedObjectType] = neighbor_agents_types[-1]

        if agent_history[-1].shape[0] == 0:
            # Return zero array when there are no agents in the scene
            agents_array = np.zeros((len(agent_history), 0, agents_states_dim))
        else:
            local_coords_agent_states = []
            padded_agent_states = _pad_agent_states(agent_history, reverse=True)
            for agent_state in padded_agent_states:
                # agent_state: np (last_frame_num_agents, 8)
                local_coords_agent_states.append(
                    convert_absolute_quantities_to_relative(
                        agent_state, ego_pose, 'agent'))
            # Calculate yaw rate
            # agents_array = (num_frames, last_Frame_num_agents, 8)
            agents_array = np.zeros(
                (len(local_coords_agent_states),
                 local_coords_agent_states[0].shape[0], agents_states_dim))

            for frame_idx in range(len(local_coords_agent_states)):
                agents_array[frame_idx, :, 0] = local_coords_agent_states[
                    frame_idx][:, AgentInternalIndex.x()].squeeze()
                agents_array[frame_idx, :, 1] = local_coords_agent_states[
                    frame_idx][:, AgentInternalIndex.y()].squeeze()
                agents_array[frame_idx, :, 2] = np.cos(
                    local_coords_agent_states[frame_idx]
                    [:, AgentInternalIndex.heading()].squeeze())
                agents_array[frame_idx, :, 3] = np.sin(
                    local_coords_agent_states[frame_idx]
                    [:, AgentInternalIndex.heading()].squeeze())
                agents_array[frame_idx, :, 4] = local_coords_agent_states[
                    frame_idx][:, AgentInternalIndex.vx()].squeeze()
                agents_array[frame_idx, :, 5] = local_coords_agent_states[
                    frame_idx][:, AgentInternalIndex.vy()].squeeze()
                agents_array[frame_idx, :, 6] = local_coords_agent_states[
                    frame_idx][:, AgentInternalIndex.width()].squeeze()
                agents_array[frame_idx, :, 7] = local_coords_agent_states[
                    frame_idx][:, AgentInternalIndex.length()].squeeze()
        # Initialize the result array
        # agents_array: (num_frames, last_Frame_num_agents, 8)
        agents = np.zeros((self.num_agents, agents_array.shape[0],
                           agents_array.shape[-1] + 3),
                          dtype=np.float32)  # (num_agents=32, num_frames, 11)
        # agents_array: (num_frames, last_Frame_num_agents, 8)
        # distance_to_ego: (last_Frame_num_agents,)
        distance_to_ego = np.linalg.norm(agents_array[-1, :, :2],
                                         axis=-1)  # (last_Frame_num_agents, )

        # Sort indices by distance
        sorted_indices = np.argsort(distance_to_ego)  # (last_Frame_num_agents)

        # Collect the indices of pedestrians and bicycles
        ped_bike_indices = [
            i for i in sorted_indices
            if agent_types[i] in (TrackedObjectType.PEDESTRIAN,
                                  TrackedObjectType.BICYCLE)
        ]  # len(ped_bike_indices) = num_pedestrian + num_bicycle
        vehicle_indices = [
            i for i in sorted_indices
            if agent_types[i] == TrackedObjectType.VEHICLE
        ]  # len(vehicle_indices) = num_vehicle
        # If the total number of available agents is less than or equal to num_agents, no need to filter further
        if len(ped_bike_indices) + len(vehicle_indices) <= self.num_agents:
            selected_indices = sorted_indices[:self.num_agents]
        else:
            # Limit the number of pedestrians and bicycles to max_ped_bike, while retaining the remaining ones for later use
            selected_ped_bike_indices = ped_bike_indices[:self.max_ped_bike]
            remaining_ped_bike_indices = ped_bike_indices[self.max_ped_bike:]

            # Combine the limited pedestrians/bicycles and all available vehicles
            selected_indices = selected_ped_bike_indices + vehicle_indices

            # If the combined selection is still less than num_agents, fill the remaining slots with additional pedestrians and bicycles
            remaining_slots = self.num_agents - len(selected_indices)
            if remaining_slots > 0:
                selected_indices += remaining_ped_bike_indices[:remaining_slots]

            # Sort and limit the selected indices to num_agents
            selected_indices = sorted(
                selected_indices,
                key=lambda idx: distance_to_ego[idx])[:self.num_agents]

        # Populate the final agents array with the selected agents' features
        for i, sorted_idx in enumerate(
                selected_indices
        ):  # selected_indices:  (shrinked_num_agents = 10)
            # agents # (num_agents=32, num_frames, 11)
            # agents_array # (num_frames, last_Frame_num_agents, 8)
            agents[
                i, :, :agents_array.
                shape[-1]] = agents_array[:,
                                          sorted_idx, :agents_array.shape[-1]]
            if agent_types[sorted_idx] == TrackedObjectType.VEHICLE:
                agents[i, :, agents_array.shape[-1]:] = [1, 0,
                                                         0]  # Mark as VEHICLE
            elif agent_types[sorted_idx] == TrackedObjectType.PEDESTRIAN:
                agents[i, :,
                       agents_array.shape[-1]:] = [0, 1,
                                                   0]  # Mark as PEDESTRIAN
            else:  # TrackedObjectType.BICYCLE
                agents[i, :, agents_array.shape[-1]:] = [0, 0,
                                                         1]  # Mark as BICYCLE
        return agents

    def _get_float_id_for_vehicle(self, v_id_str: str) -> float:
        if v_id_str not in self.str_id_to_int_id:
            self.str_id_to_int_id[v_id_str] = self._next_int_id
            self._next_int_id += 1
        return float(self.str_id_to_int_id[v_id_str])

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        # TODO: reset 해도 id를 유지해야 하는 부분이 있는지 궁금
        self.str_id_to_int_id.clear()
        self._next_int_id = 1
        self.history_buffer.reset()
        super(HistoricalBufferTrafficManager, self).reset()
        # neighbor_agents, neighbor_types = self._collect_neighbor_data()
        # self.history_buffer.update(neighbor_agents, neighbor_types)

    def after_step(self, *args, **kwargs):
        super(HistoricalBufferTrafficManager, self).after_step(*args, **kwargs)
        neighbor_agents, neighbor_types = self._collect_neighbor_data()
        self.history_buffer.update(neighbor_agents, neighbor_types)

        return dict()

    def _collect_neighbor_data(self) -> (np.ndarray, List[TrackedObjectType]):
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
            # toy example
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
            type_list.append(TrackedObjectType.VEHICLE)

        arr = np.array(data_list, dtype=np.float32)
        return arr, type_list


# For compatibility check
TrafficManager = PGTrafficManager

# ── new_mixed_traffic_manager.py ───────────────────────────────────────────────


def rotation_matrix(theta: float) -> np.ndarray:
    """주어진 θ로부터 2×2 회전 행렬 반환."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def ego_to_global(traj_ego: np.ndarray, ego_pos: np.ndarray,
                  ego_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """
    traj_ego: (T,4) array of [x_ego, y_ego, cos_ego_yaw, sin_ego_yaw]
    ego_pos: (2,) global 위치
    ego_yaw: 스칼라 ego yaw
    returns:
      coords_global: (T,2) global x,y
      yaw_global:   (T,) global yaw
    """
    coords_ego = traj_ego[:, :2]  # (T,2)
    yaw_ego_frame = np.arctan2(traj_ego[:, 3], traj_ego[:, 2])  # (T,)
    R_e2g = rotation_matrix(ego_yaw)  # ego→global 회전
    coords_global = coords_ego.dot(R_e2g.T) + ego_pos  # (T,2)
    yaw_global = yaw_ego_frame + ego_yaw  # (T,)
    return coords_global, yaw_global


def global_to_local(coords_global: np.ndarray, yaw_global: np.ndarray,
                    veh_pos: np.ndarray, veh_yaw: float) -> np.ndarray:
    """
    coords_global: (T,2), yaw_global: (T,)
    veh_pos: (2,), veh_yaw: 스칼라
    returns:
      future_traj: (T,4) array of [x_local, y_local, cos_local_yaw, sin_local_yaw]
    """
    R_g2v = rotation_matrix(-veh_yaw)  # global→veh 회전
    delta = coords_global - veh_pos  # (T,2)
    coords_local = delta.dot(R_g2v.T)  # (T,2)
    yaw_local = yaw_global - veh_yaw  # (T,)

    cos_l = np.cos(yaw_local)[:, None]  # (T,1)
    sin_l = np.sin(yaw_local)[:, None]  # (T,1)
    return np.concatenate([coords_local, cos_l, sin_l], axis=1)


def transform_trajectory(npc_traj_wrt_ego: np.ndarray, ego_pos: np.ndarray,
                         ego_yaw: float, veh_pos: np.ndarray,
                         veh_yaw: float) -> np.ndarray:
    """
    ego계 기준 npc_traj_wrt_ego → 각 vehicle 로컬계 기준 (T,4) trajectory.
    """
    coords_g, yaw_g = ego_to_global(npc_traj_wrt_ego, ego_pos, ego_yaw)
    return global_to_local(coords_g, yaw_g, veh_pos, veh_yaw)


class DiffusionTrafficManager(HistoricalBufferTrafficManager):

    def __init__(self):
        super().__init__()
        # 시각화된 궤적 NodePath를 보관할 리스트
        self._traffic_traj_nodes: list = []

    def reset(self):
        # 기존 reset 처리
        super().reset()
        # 궤적 그림 초기화
        self._clear_traffic_trajs()
        # safety cap: 항상 최대 10대
        # TODO: remove
        if len(self._traffic_vehicles) > 10:
            self._traffic_vehicles = self._traffic_vehicles[:10]

    def _clear_traffic_trajs(self):
        """이전 프레임에 그렸던 궤적 NodePath를 모두 제거."""
        for np_node in self._traffic_traj_nodes:
            np_node.removeNode()
        self._traffic_traj_nodes.clear()

    def _draw_all_traffic_trajs(self):
        """
        engine.external_npc_actions((N, T, 4): x,y,cos(yaw),sin(yaw))를
        ego→global 변환 후, 각 차량 위치 궤적을 월드에 그린다.
        """
        engine = self.engine
        # active ego 위치/방향
        ego = next(iter(engine.agent_manager.active_agents.values()))
        ego_pos = np.array([ego.rear_axle_xy[0], ego.rear_axle_xy[1]])
        ego_yaw = ego.heading_theta

        # 외부 NPC들이 예측해온 궤적
        external_npc = engine.external_npc_actions  # [:, :1, :]  # (N, T, 4)
        # 각 traffic 차량의 글로벌 궤적 좌표 구하기
        # 3) 차량별로 한 궤적씩 변환 → world coords (T,2)
        for idx, npc_traj in enumerate(external_npc):  # npc_traj.shape == (T,4)
            # 만약 npc_traj 의 값이 전부 0이라면, skip
            if np.all(npc_traj == 0.):
                continue
            coords_g, yaws_g = ego_to_global(npc_traj, ego_pos, ego_yaw)
            # 4) 각 점을 월드에 짧은 선으로 찍기
            for (x, y) in coords_g:
                np_node = engine._draw_line_3d(
                    LVector3(x, y, 1.5),
                    LVector3(x, y, (idx + 1.) * 3. + 2.),
                    color=(0, 1, 0, 1),  # 초록
                    thickness=250)
                np_node.reparentTo(engine.render)
                self._traffic_traj_nodes.append(np_node)

    # ────────────────────────────────────────────────────────────────────────
    # reset 단계 – 트래픽 차량을 만든 뒤 1회 초기 분배
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # 매 step 직전 먼저 policy 구성을 갱신한 뒤, 부모 로직 수행
    # ────────────────────────────────────────────────────────────────────────
    def before_step(self):
        """
        1) block trigger → 새 traffic 차량을 self._traffic_vehicles 에 등록
        2) 가장 가까운 10 대에 LQRPolicy 부여
        3) 각 vehicle.before_step(action) 호출
        """
        # self._clear_traffic_trajs()
        # self._draw_all_traffic_trajs()
        external_npc_actions = self.engine.external_npc_actions[:,
                                                                1:]  # (P-1, 80, 4)
        diffusion_vehicle_num = external_npc_actions.shape[0]
        # ── 2.  이제 리스트가 확정됐으므로 policy 재배치
        closest_idx = self._update_control_policies(0)
        # (2) Ego 정보 한 번만 꺼내두기
        ego = next(iter(self.engine.agent_manager.active_agents.values()))
        ego_pos = np.array(ego.position[:2], dtype=np.float32)
        ego_yaw = ego.heading_theta
        if closest_idx is not None:
            sorted_traffic_vehicles = [
                self._traffic_vehicles[i] for i in closest_idx
            ]
            for vehicle_idx, veh in enumerate(sorted_traffic_vehicles):
                pol = self.engine.get_policy(veh.id)
                assert isinstance(pol, (LQRPolicy))
                npc_traj_wrt_ego = external_npc_actions[vehicle_idx]
                # global → vehicle 로컬로 일괄 변환
                future_traj = transform_trajectory(
                    npc_traj_wrt_ego, ego_pos, ego_yaw,
                    np.array(veh.position[:2], dtype=np.float32),
                    veh.heading_theta)
                veh.before_step(pol.act(veh.id, future_traj))

        # ── 1.  block trigger 처리 (부모 로직 그대로)
        if self.mode != TrafficMode.Respawn:
            for ego in self.engine.agent_manager.active_agents.values():
                if self.block_triggered_vehicles:
                    ego_road = Road(ego.lane_index[0], ego.lane_index[1])
                    if ego_road == self.block_triggered_vehicles[
                            -1].trigger_road:
                        blk = self.block_triggered_vehicles.pop()
                        self._traffic_vehicles += list(
                            self.get_objects(blk.vehicles).values())

        for vehicle_idx, veh in enumerate(self._traffic_vehicles):
            pol = self.engine.get_policy(veh.id)
            if isinstance(pol, IDMPolicy):
                veh.before_step(pol.act(is_kinematic=True))

        #
        # # ── 3.  action 적용
        # for vehicle_idx, veh in enumerate(self._traffic_vehicles):
        #     pol = self.engine.get_policy(veh.id)
        #     if isinstance(pol, IDMPolicy):
        #         veh.before_step(pol.act(is_kinematic=True))
        #     elif isinstance(pol, LQRPolicy):
        #         index = np.where(closest_idx == vehicle_idx)[0][0]
        #         future_trajectory = external_npc_actions[index] # (80, 4)
        #         veh.before_step(pol.act(veh.id, future_trajectory))
        return {}

    # ────────────────────────────────────────────────────────────────────────
    # 내부 : 현 시점 traffic 차량들에 대해 “가까운 11대” 재계산 → policy 교체
    # ────────────────────────────────────────────────────────────────────────
    def _update_control_policies(self, diffusion_vehicle_num=10) -> np.ndarray:
        if not self._traffic_vehicles or diffusion_vehicle_num == 0:  # ── (0) early-return
            return None

        # ── (1) 대표 ego 선정 ──────────────────────────────────────────────
        ego_list = list(self.engine.agent_manager.active_agents.values())
        if not ego_list:
            raise RuntimeError("No active agents found in the scene.")
        ego_pos = ego_list[0].position  # (x, y, z) 또는 (x, y)

        # ── (2) 벡터화 거리 계산 & 11대 선별 ───────────────────────────────
        #      ->  Python loop 대신 NumPy C-루틴: GIL 해제 + SIMD 가능
        veh_positions = np.asarray([v.position for v in self._traffic_vehicles],
                                   dtype=np.float32)  # (N, 2/3)
        dists = np.linalg.norm(veh_positions - ego_pos, axis=1)  # (N,)

        diffusion_vehicle_num = min(diffusion_vehicle_num, len(dists))
        closest_idx = np.argsort(dists)[:diffusion_vehicle_num]
        # closest_idx = np.argpartition(
        #     dists, diffusion_vehicle_num)[:diffusion_vehicle_num]  # k 개 인덱스
        lqr_target_set = {self._traffic_vehicles[i] for i in closest_idx}

        # ── (3) 교체가 필요한 차량만 따로 모아 한 번에 처리 ────────────────
        swap_cache = []  # (veh, desired_cls)
        get_policy = self.engine.get_policy
        for veh in self._traffic_vehicles:
            desired_cls = LQRPolicy if veh in lqr_target_set else IDMPolicy
            current_pol = get_policy(veh.id)
            if current_pol is None or not isinstance(current_pol, desired_cls):
                swap_cache.append((veh, desired_cls))

        # 실제 엔진 state 를 바꾸는 작업은 최소 loop 로
        for veh, cls in swap_cache:
            # engine.add_policy → BasePolicy(control_object, random_seed, …)
            self.add_policy(veh.id, cls, veh, self.generate_seed())
        return closest_idx

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random,
                                           p=[0.2, 0.3, 0.3, 0.2, 0.0],
                                           vehicle_type="bicycle_history")
        return vehicle_type
