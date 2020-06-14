import numpy as np

from smarts.zoo.registry import register, make
from smarts.core.agent_interface import AgentInterface, AgentType

from smarts.core.controllers import ActionSpaceType
from smarts.env.agent import Agent, AgentPolicy


class KeeplanePolicy(AgentPolicy):
    def act(self, obs):
        # return "change_right"
        return "keep_lane"


class RandomPolicy(AgentPolicy):
    def __init__(self):
        self.action_choices = [
            "keep_lane",
            "slow_down",
            "change_lane_left",
            "change_lane_right",
        ]

    def act(self, obs):
        return np.random.choice(self.action_choices)


# to save time
keeplane_agent = Agent(
    interface=AgentInterface(
        max_episode_steps=None,
        waypoints=True,
        neighborhood_vehicles=False,
        ogm=False,
        rgb=False,
        lidar=False,
        action=ActionSpaceType.Lane,
    ),
    # interface=AgentInterface.from_type(
    #     AgentType.Laner, max_episode_steps=2000
    # ),
    policy=KeeplanePolicy(),
)

random_agent = Agent(
    interface=AgentInterface(
        max_episode_steps=None,
        waypoints=True,
        neighborhood_vehicles=False,
        ogm=False,
        rgb=False,
        lidar=False,
        action=ActionSpaceType.Lane,
    ),
    policy=RandomPolicy(),
)


# keep lane social agent
register(
    locator="zoo-agent1-v0", entry_point=lambda: keeplane_agent,
)

# random act social agent
register(
    locator="zoo-agent2-v0", entry_point=lambda: random_agent,
)
