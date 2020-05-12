import os
import sys
from pathlib import Path

import gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.custom_observations import lane_ttc_observation_adapter
from smarts.env.agent import Agent, AgentPolicy

tf = try_import_tf()


# This action space should match the input to the action(..) function below.
ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)


# This observation space should match the output of observation_adapter(..) below
OBSERVATION_SPACE = lane_ttc_observation_adapter.space


def observation_adapter(env_observation):
    return lane_ttc_observation_adapter.transform(env_observation)


def reward_adapter(env_obs, env_reward):
    return env_reward


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])


class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"


ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


class ModelPolicy(AgentPolicy):
    def __init__(self, path_to_model, observation_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._path_to_model = path_to_model

    def setup(self):
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        tf.saved_model.load(self._sess, export_dir=self._path_to_model, tags=["serve"])

    def teardown(self):
        # TODO: figure out the exact params to pass to this __exit__ call
        # self._sess.__exit__()
        pass

    def act(self, obs):
        obs = self._prep.transform(obs)
        graph = tf.get_default_graph()
        # These tensor names were found by inspecting the trained model
        output_node = graph.get_tensor_by_name("default_policy/add:0")
        input_node = graph.get_tensor_by_name("default_policy/observation:0")
        res = self._sess.run(output_node, feed_dict={input_node: [obs]})
        action = res[0]
        return action


model_path = Path(__file__).parent / "model"

agent = Agent(
    interface=AgentInterface.from_type(
        AgentType.StandardWithAbsoluteSteering, max_episode_steps=1000
    ),
    policy=ModelPolicy(str(model_path.absolute()), OBSERVATION_SPACE,),
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
)
