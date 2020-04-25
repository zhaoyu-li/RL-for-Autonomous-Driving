import random
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.agent import Agent, AgentPolicy


class Policy(AgentPolicy):
    def act(self, obs):
        return np.array(
            [random.uniform(0, 1), random.uniform(0, 1), random.uniform(-1, 1)]
        )


agent = Agent(
    interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=2000),
    policy=Policy(),
)
