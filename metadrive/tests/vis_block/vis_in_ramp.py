"""This file visualizes a InRampOnStraight block. Use mouse left button to draw down for zooming out."""
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.ramp import InRampOnStraight
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":

    test = TestBlock()

    initialize_asset_loader(test)

    global_network = NodeRoadNetwork()
    straight = FirstPGBlock(global_network, 3.0, 1, test.render, test.world, 1)
    straight = Straight(4, straight.get_socket(0), global_network, 1)
    straight.construct_block(test.render, test.world)
    # print(len(straight.dynamic_nodes))
    for i in range(1, 3):
        straight = InRampOnStraight(i, straight.get_socket(0), global_network,
                                    i)
        straight.construct_block(test.render, test.world)
        # print(len(straight.dynamic_nodes))
    test.show_bounding_box(global_network)
    test.run()
