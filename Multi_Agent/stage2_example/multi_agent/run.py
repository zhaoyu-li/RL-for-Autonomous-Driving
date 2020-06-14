# this file is for evaluation
import os
import gym
from pathlib import Path
from agent import agent
from smarts.core import scenario


scenario_dir = (Path(__file__).parent / "f1_public").resolve()
scenario_names = ["shanghai", "silverstone", "monte", "interlagos"]
scenario_paths = [scenario_dir / name for name in scenario_names]

AGENT_ID = "Agent-007"


def main():
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenario_paths,
        agents={AGENT_ID: agent},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=42,
    )

    agent.reset()

    while True:
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
            total_reward += rewards[AGENT_ID]
        print("Accumulated reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()
