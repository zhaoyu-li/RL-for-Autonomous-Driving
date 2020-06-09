import os
import random
import argparse

import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

import smarts
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.utils import copy_tree

from agent import agent, TrainingModel


# Path to the scenario to test
scenario_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../dataset_public/3lane"
)


def on_episode_start(info):
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["ego_speed"] = []


def on_episode_step(info):
    episode = info["episode"]
    single_agent_id = list(episode._agent_to_last_obs)[0]
    obs = episode.last_raw_obs_for(single_agent_id)
    episode.user_data["ego_speed"].append(obs["speed"])


def on_episode_end(info):
    episode = info["episode"]
    mean_ego_speed = np.mean(episode.user_data["ego_speed"])

    episode.custom_metrics["mean_ego_speed"] = mean_ego_speed

    agent_scores = []
    for info in episode._agent_to_last_info.values():
        agent_scores.append(info["score"])
    mean_dis = np.mean(agent_scores)
    episode.custom_metrics["distance_travelled"] = mean_dis

    print(
        "episode {} ended with length {}, distance {}, and mean ego speed {:.2f}".format(
            episode.episode_id, episode.length, mean_dis, mean_ego_speed
        )
    )


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


def main(args):
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=300,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore,
    )

    # XXX: There is a bug in Ray where we can only export a trained model if
    #      the policy it's attached to is named 'default_policy'.
    #      See: https://github.com/ray-project/ray/issues/5339
    rllib_policies = {
        "default_policy": (
            None,
            agent.observation_space,
            agent.action_space,
            {"model": {"custom_model": TrainingModel.NAME}},
        )
    }

    smarts.core.seed(args.seed)
    tune_config = {
        "env": RLlibHiWayEnv,
        "log_level": "WARN",
        "num_workers": args.num_workers,
        "env_config": {
            "seed": tune.sample_from(lambda spec: random.randint(0, 300)),
            "scenarios": [scenario_path],
            "headless": args.headless,
            "agents": {f"AGENT-{i}": agent for i in range(args.num_agents)},
        },
        "multiagent": {"policies": rllib_policies},
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
    }

    experiment_name = "rllib_pbt_example"
    log_dir = os.path.expanduser("~/ray_results")

    result_dir = args.result_dir
    checkpoint = None
    if args.checkpoint_num is not None:
        checkpoint = f"{result_dir}/checkpoint_{args.checkpoint_num}/checkpoint-{args.checkpoint_num}"

    print(f"Checkpointing at {log_dir}")
    analysis = tune.run(
        "PPO",
        name=experiment_name,
        stop={"time_total_s": 6 * 60 * 60},  # 6 hour
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir=log_dir,
        resume=args.resume_training,
        restore=checkpoint,
        max_failures=1000,
        num_samples=args.num_samples,
        export_formats=["model", "checkpoint"],
        config=tune_config,
        scheduler=pbt,
    )

    print(analysis.dataframe().head())

    logdir = analysis.get_best_logdir("episode_reward_max")
    model_path = os.path.join(logdir, "model")
    dest_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")

    copy_tree(model_path, dest_model_path, overwrite=True)
    print(f"Wrote model to: {dest_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rllib-example")
    parser.add_argument(
        "--headless", help="run simulation in headless mode", action="store_true",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of times to sample from hyperparameter space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The base random seed to use, intended to be mixed with --num_samples",
    )
    parser.add_argument(
        "--num_agents", type=int, default=1, help="Number of agents (one per policy)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers used to sample"
    )
    parser.add_argument(
        "--resume_training",
        default=False,
        action="store_true",
        help="Resume the last trained example",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/home/ray_results",
        help="Directory containing results",
    )
    parser.add_argument(
        "--checkpoint_num", type=int, default=None, help="Checkpoint number"
    )
    args = parser.parse_args()
    main(args)
