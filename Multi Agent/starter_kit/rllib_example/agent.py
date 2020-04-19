import os
import sys
from pathlib import Path

import gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.custom_observations import ttc_by_path
from smarts.env.agent import Agent, AgentPolicy

tf = try_import_tf()


# This action space should match the input to the action(..) function below.
ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)


# This observation space should match the output of observation(..) below
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-180, high=180, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def ego_ttc_calc(ego, ego_lane_index, ttc_by_path, lane_dist_by_path):
    ego_ttc = [0] * 3
    ego_lane_dist = [0] * 3

    ego_ttc[1] = ttc_by_path[ego_lane_index]
    ego_lane_dist[1] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path) - 1
    min_lane_index = 0
    if ego_lane_index + 1 > max_lane_index:
        ego_ttc[2] = 0
        ego_lane_dist[2] = 0
    else:
        ego_ttc[2] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[2] = lane_dist_by_path[ego_lane_index + 1]
    if ego_lane_index - 1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 1]
    return ego_ttc, ego_lane_dist


def observation_adapter(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    ttc_by_p, lane_dist_by_p = ttc_by_path(
        ego, waypoint_paths, env_observation.neighborhood_vehicle_states
    )
    ego_ttc, ego_lane_dist = ego_ttc_calc(
        ego, closest_wp.lane_index, ttc_by_p, lane_dist_by_p
    )

    return {
        "distance_from_center": np.array([norm_dist_from_center]),
        "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
        "speed": np.array([ego.speed]),
        "steering": np.array([ego.steering]),
        "ego_ttc": np.array(ego_ttc),
        "ego_lane_dist": np.array(ego_lane_dist),
    }


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
    interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=1000),
    policy=ModelPolicy(str(model_path.absolute()), OBSERVATION_SPACE,),
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
)
