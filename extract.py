import torch
import numpy as np
import matplotlib.pyplot as plt

# 1) Load the data
loaded = torch.load('observation.pt')
# for k,v in loaded.items():
#     print(k, v.shape)
a = loaded["lanes_speed_limit"] * 3.6
print(a)
# neighbor_agents_past = loaded['neighbor_agents_past'][0, :, :, :2].cpu().numpy()  # (32,21,2)
#
# future = torch.load('denorm_dit_output.pt')
# future_trajectories = future[0, :, :, :2].cpu().numpy()  # (11,81,2)
#
# # 2) Plot everything
# plt.figure(figsize=(8, 8))
# plt.title("Neighbor Agents Past (black) vs. Future Trajectories (red)")
# plt.xlabel("x")
# plt.ylabel("y")
#
# # Black: history of each neighbor
# for traj in neighbor_agents_past:
#     plt.plot(traj[:, 0], traj[:, 1], color='black', linewidth=1, alpha=0.7)
#
# # Red: predicted futures
# for traj in future_trajectories:
#     plt.plot(traj[:, 0], traj[:, 1], color='red', linewidth=2)
#
# plt.axis('equal')
# plt.grid(True)
# plt.show()
