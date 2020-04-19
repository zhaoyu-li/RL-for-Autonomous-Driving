import os
import gym
from pathlib import Path
from agent import agent
from smarts.core import scenario

# Path to the scenario to test
scenario_path = (Path(__file__).parent / "../dataset_public/3lane").resolve()

AGENT_ID = "Agent-007"


def main():
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agents={AGENT_ID: agent},
        envision=True,
        headless=True,
        visdom=False,
        seed=42,
    )
    agent.reset()
    observations = env.reset()

    total_reward = 0.0
    dones = {"__all__": False}
    while not dones["__all__"]:
        agent_obs = observations.agent_observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
        total_reward += rewards.agent_rewards[AGENT_ID]

    env.close()

    print("Accumulated reward:", total_reward)


if __name__ == "__main__":
    main()
