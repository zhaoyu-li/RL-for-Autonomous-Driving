from pathlib import Path

import ray
from agent import agent

from smarts.env.rllib_hiway_env import RLlibHiWayEnv

# Path to the scenario to test
scenario_path = (Path(__file__).parent / "../dataset_public/3lane").resolve()

AGENT_ID = "Agent-007"


def main():
    ray.init()

    env = RLlibHiWayEnv(
        config={
            "seed": 42,
            "scenarios": [scenario_path],
            "agents": {AGENT_ID: agent},
            "headless": False,
            "visdom": False,
        }
    )

    agent.reset()
    observations = env.reset()

    total_reward = 0.0
    dones = {"__all__": False}
    while not dones["__all__"]:
        agent_obs = observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
        total_reward += rewards[AGENT_ID]

    env.close()

    print("Accumulated reward:", total_reward)


if __name__ == "__main__":
    main()
