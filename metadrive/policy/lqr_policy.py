from metadrive.policy.base_policy import BasePolicy

class LQRPolicy(BasePolicy):
    """
    LQR policy for controlling the vehicle in the simulation environment.
    This policy uses a Linear Quadratic Regulator (LQR)
    to compute the control inputs based on the current state of the vehicle.
    """

    def __init__(self, control_object):
        super().__init__(control_object)

    def act(self, agent_id):
        future_trajectory = self.engine.external_actions[agent_id]
        # future_trajectory: np.ndarray (80, 4) # 80: number of future steps, 4: x, y, vx, vy
        # Convert future_trajectory to control inputs(acceleration and steering rate)


