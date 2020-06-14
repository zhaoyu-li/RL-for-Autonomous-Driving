# this file is for training and evaluation
import os
import pickle
from pathlib import Path

import gym
import numpy as np
from custom_observations import lane_ttc_observation_adapter
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.agent import Agent, AgentPolicy

tf = try_import_tf()

#########################################
# Spaces and Adpaters
#########################################

# This action space should match the input to the action(..) function below.
# continuous action space
ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)

# discrete action space
# ACTION_SPACE = gym.spaces.Discrete(4)

# This observation space should match the output of observation_adapter(..) below
OBSERVATION_SPACE = lane_ttc_observation_adapter.space


def observation_adapter(env_observation):
    return lane_ttc_observation_adapter.transform(env_observation)


def reward_adapter(env_obs, env_reward):
    return env_reward


# continuous
def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])


# discrete
""""
def action_adapter(model_action):
    # Also action can be this "{'lane_change':-1,'target_speed':50}" ...
    action_list = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    action = action_list[model_action]
    return action
"""


class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"


ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


#########################################
# restore checkpoint generated during training, like checkpoint_66/checkpoint-66.
#########################################
class RLlibTFCheckpointPolicy(AgentPolicy):
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        self._load_path = load_path
        self._algorithm = algorithm
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

    def setup(self):
        if self._sess:
            return

        if self._algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif self._algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif self._algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif self._algorithm == "DQN":
            from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy as LoadPolicy
        else:
            raise TypeError("Unsupport algorithm")

        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        with tf.name_scope(self._policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            self.policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(self._load_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[self._policy_name]
            self.policy.set_weights(weights)

    def teardown(self):
        # TODO: actually teardown the TF session
        pass

    def act(self, obs):
        obs = self._prep.transform(obs)
        action = self.policy.compute_actions([obs], explore=False)[0][0]

        return action


#########################################
# restore checkpoint exported at end, like checkpoint/checkpoint
#########################################
class RLlibFinalCkptPolicy(AgentPolicy):
    def __init__(self, path_to_model, observation_space, action_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._path_to_model = path_to_model

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

    def setup(self):
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        saver = tf.train.import_meta_graph(
            os.path.join(os.path.dirname(self._path_to_model), "model.meta")
        )
        saver.restore(
            self._sess, os.path.join(os.path.dirname(self._path_to_model), "model")
        )

        graph = tf.get_default_graph()

        if self.is_continuous:
            # These tensor names were found by inspecting the trained model
            # deterministic
            self.output_node = graph.get_tensor_by_name("default_policy/split:0")
            # add guassian noise
            # output_node = graph.get_tensor_by_name("default_policy/add:0")
        else:
            self.output_node = graph.get_tensor_by_name("default_policy/ArgMax:0")

        self.input_node = graph.get_tensor_by_name("default_policy/observation:0")

    def teardown(self):
        # TODO: figure out the exact params to pass to this __exit__ call
        # self._sess.__exit__()
        pass

    def act(self, obs):
        obs = self._prep.transform(obs)
        res = self._sess.run(self.output_node, feed_dict={self.input_node: [obs]})
        action = res[0]
        return action


#########################################
# restore model exported at end, like model/
#########################################
class RLlibModelPolicy(AgentPolicy):
    def __init__(self, path_to_model, observation_space, action_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._path_to_model = path_to_model

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

    def setup(self):
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        tf.saved_model.load(self._sess, export_dir=self._path_to_model, tags=["serve"])

        graph = tf.get_default_graph()

        if self.is_continuous:
            # These tensor names were found by inspecting the trained model
            # deterministic
            self.output_node = graph.get_tensor_by_name("default_policy/split:0")
            # add guassian noise
            # output_node = graph.get_tensor_by_name("default_policy/add:0")
        else:
            self.output_node = graph.get_tensor_by_name("default_policy/ArgMax:0")

        self.input_node = graph.get_tensor_by_name("default_policy/observation:0")

    def teardown(self):
        # TODO: figure out the exact params to pass to this __exit__ call
        # self._sess.__exit__()
        pass

    def act(self, obs):
        obs = self._prep.transform(obs)
        res = self._sess.run(self.output_node, feed_dict={self.input_node: [obs]})
        action = res[0]
        return action


model_path = Path(__file__).parent / "model"
# model_path = Path(__file__).parent / "checkpoint_200/checkpoint-200"
# model_path = Path(__file__).parent / "checkpoint/checkpoint"


#########################################
# continous action space agent
#########################################
# custom agent interface example
"""
agent_interface = AgentInterface(
    max_episode_steps=None,
    waypoints=True,
    neighborhood_vehicles=NeighborhoodVehicles(100),
    ogm=OGM(64, 64, 0.25),
    rgb=False,
    lidar=False,
    action=ActionSpaceType.Continuous,
)
"""

agent = Agent(
    interface=AgentInterface.from_type(
        AgentType.StandardWithAbsoluteSteering, max_episode_steps=2000
    ),
    policy=RLlibModelPolicy(
        str(model_path.absolute()), OBSERVATION_SPACE, ACTION_SPACE
    ),
    # policy=RLlibTFCheckpointPolicy(str(model_path.absolute()), "PPO",   "default_policy", OBSERVATION_SPACE, ACTION_SPACE),
    # policy=RLlibFinalCkptPolicy(str(model_path.absolute()), OBSERVATION_SPACE, ACTION_SPACE),
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
)

#########################################
# discrete action space agent
#########################################
"""
agent = Agent(
    interface=AgentInterface.from_type(
        AgentType.Laner, max_episode_steps=2000
    ),
    policy=RLlibModelPolicy(str(model_path.absolute()), OBSERVATION_SPACE, ),
    # policy=RLlibTFCheckpointPolicy(str(model_path.absolute()), "PPO",   "default_policy", OBSERVATION_SPACE, ACTION_SPACE),
    # policy=RLlibFinalCkptPolicy(str(model_path.absolute()), OBSERVATION_SPACE, ),
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
)
"""
