from metadrive.envs.metadrive_env import MetaDriveEnv


def test_traffic_mode(render=False):
    try:
        for mode in ["hybrid", "trigger", "respawn"]:
            env = MetaDriveEnv({
                "num_scenarios": 1,
                "traffic_density": 0.1,
                "traffic_mode": mode,
                "start_seed": 22,
                "use_render": render,
                "map": "X",
            })

            o, _ = env.reset()
            env.agent.set_velocity([1, 0.1], 10)
            if mode == "respawn":
                assert len(env.engine.traffic_manager._traffic_vehicles) != 0
            elif mode == "hybrid" or mode == "trigger":
                assert len(env.engine.traffic_manager._traffic_vehicles) == 0
            for s in range(1, 300):
                o, r, tm, tc, info = env.step(env.action_space.sample())
            if mode == "hybrid" or mode == "respawn":
                assert len(env.engine.traffic_manager._traffic_vehicles) != 0
            elif mode == "trigger":
                assert len(env.engine.traffic_manager._traffic_vehicles) == 0
            env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_traffic_mode()
