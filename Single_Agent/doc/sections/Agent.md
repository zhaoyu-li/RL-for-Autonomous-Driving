## Agent

SMARTS provides users the ability to custom their agents. `Agent` class has the following fields: 

```python
class Agent:
    interface: AgentInterface
    policy: Union[AgentPolicy, Callable] = None
    observation_space: gym.Space = None
    action_space: gym.Space = None
    observation_adapter: Callable = default_obs_adapter
    reward_adapter: Callable = default_reward_adapter
    action_adapter: Callable = default_action_adapter
    info_adapter: Callable = default_info_adapter
```

An example of how to create an `Agent` instance is shown below.
```python
Agent(
    interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)
```

We will further explain the fields of `Agent` class later on this page. You can also read the source code at `smarts/env/agent.py`.

### AgentInterface

`AgentInterface` regulates the flow of informatoin between the an agent and a SMARTS environment. It specifies the observations an agent expects to receive from the environment and the action the agent does to the environment. To create an agent interface, you can try

```python
agent_interface = AgentInterface.from_type(
    interface = AgentType.Standard,
    max_episode_steps = 1000, 
    ...
)
```

SMARTS provide some interface types, and the differences between them is shown in the table below. **T** means the `AgentType` will provide this option or information. 

|                       |       AgentType.Full       | AgentType.StandardWithAbsoluteSteering |       AgentType.Standard        |   AgentType.Laner    |
| :-------------------: | :------------------------: | :------------------------------------: | :-----------------------------: | :------------------: |
|   max_episode_steps   |           **T**            |                 **T**                  |              **T**              |        **T**         |
| neighborhood_vehicles |           **T**            |                 **T**                  |              **T**              |                      |
|       waypoints       |           **T**            |                 **T**                  |              **T**              |        **T**         |
|          ogm          |           **T**            |                                        |                                 |                      |
|          rgb          |           **T**            |                                        |                                 |                      |
|         lidar         |           **T**            |                                        |                                 |                      |
|        action         | ActionSpaceType.Continuous |       ActionSpaceType.Continuous       | ActionSpaceType.ActuatorDynamic | ActionSpaceType.Lane |
|         debug         |           **T**            |                 **T**                  |              **T**              |        **T**         |

`max_episode_steps` controls the max running steps allowed for the agent in an episode. The default `None` setting means agents have no such limit.
You can move max_episode_steps control authority to RLlib with their config option `horizon`, but lose the ability to customize 
different max_episode_len for each agent.

`action` controls the agent action type used. There are three `ActionSpaceType`: ActionSpaceType.Continuous, ActionSpaceType.Lane 
and ActionSpaceType.ActuatorDynamic.
- `ActionSpaceType.Continuous`: continuous action space with throttle, brake, absolute steering angle.
- `ActionSpaceType.ActuatorDynamic`: continuous action space with throttle, brake, steering rate. Steering rate means
the amount of steering angle change *per second* (either positive or negative) to be applied to the current steering angle.
- `ActionSpaceType.Lane`: discrete lane action space of strings including "keep_lane",  "slow_down", "change_lane_left", "change_lane_right". (WARNING: This is the case in the current version 0.3.2b, but a newer version will soon be released. In this newer version, the action space will no longer being strings, but will be a tuple of an integer for `lane_change` and a float for `target_speed`.)

For other observation options, see [Observations](Observations.md) for details.

We recommend you custom your agent_interface, like
```python
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType

agent_interface = AgentInterface(
    max_episode_steps=1000,
    waypoints=True,
    neighborhood_vehicles=True,
    ogm=True,
    rgb=True,
    lidar=False,
    action=ActionSpaceType.Continuous,
)
```

For further customation, you can try
```python
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, OGM, RGB, Waypoints
from smarts.core.controllers import ActionSpaceType

agent_interface = AgentInterface(
    max_episode_steps=1000,
    waypoints=Waypoints(lookahead=50), # lookahead 50 meters
    neighborhood_vehicles=NeighborhoodVehicles(radius=50), # only get neighborhood info with 50 meters.
    ogm=True,
    rgb=True,
    lidar=False,
    action=ActionSpaceType.Continuous,
)
```
refer to `smarts/core/agent_interface` for more details.


IMPORTANT: The generation of OGM (`ogm=True`) and RGB (`rgb=True`) images may significantly slow down the environment `step()`. If your model does not consume such observations, we recommend that you set them to `False`.

IMPORTANT: Depending on how your agent model is set up, `ActionSpaceType.ActuatorDynamic` might allow the agent to learn faster than `ActionSpaceType.Continuous` simply because learning to correct steering could be simpler than learning a mapping to all the absolute steering angle values. But, again, it also depends on the design of your agent model. 

## Policy

A policy is a provider that takes in the observations of an agent and decides on an action.

```python
# A simple policy that ignores observations
class IgnoreObservationsPolicy(AgentPolicy):
    def act(self, obs):
        return [throttle, brake, steering_rate]
```
The observation passed in should be the observations that a given agent sees. In **contininuous action space** the policy is expected to pass out values for `throttle` [0->1], `brake` [0->1], and `steering_rate` [-1->1].

Otherwise, only while using **lane action space**, the policy is expected to return a laning related command: `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, `"change_lane_right"`.

The `Policy` is needed when we use `Agent` for evaluation. Otherwise it is not necessary for RL training.

## Adapters and Spaces

Adapters convert the data such as an agent's raw sensor observations to a more useful form. And spaces provide samples for variation.
 Adapters and spaces are particularly relevant to the `rllib_example/agent.py` example. Also check out `smarts/env/custom_observations.py` for some processing examples.

```python
# Adapter
def observation_adapter(env_observation):
    ego = env_observation.ego_vehicle_state

    return {
        "speed": [ego.speed],
        "steering": [ego.steering],
    }

# Associated Space
# You want to match the space to the adapter
OBSERVATION_SPACE = gym.spaces.Dict(
{
    ## see http://gym.openai.com/docs/#spaces
    "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),        
}
```

You can custom your metrics and design your own observations like `smarts/env/custom_observations.py`.
In `smarts/env/custom_observations.py`, the custom observation's meaning is:

- "distance_from_center": distance to lane center 
- "angle_error": ego heading relative to the closest waypoint
- "speed": ego speed
- "steering": ego steering
- "ego_ttc": time to collision in each lane
- "ego_lane_dist": closest cars' distance to ego in each lane


Likewise with the action adapter

```python
# this comes in from the output of the Policy
def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)
```

Because the reward is just a scalar value, no explicit specification of space is needed to go with the reward adapter. But the reward adapter is very important because it  allows further shaping of the reward to your liking:

```python
    def reward_adapter(env_obs, env_reward):
        return env_reward
```

Similarly, the info adapter allows further processing on the extra info, if you somehow need that.
```python
def info_adapter(env_obs, env_reward, env_info):
    env_info[INFO_EXTRA_KEY] = "blah"
    return env_info
```

## Agent Observations

Of all the information to work with it is useful to know a bit about the main agent observations in particular.

For that see the [Observations](Observations.md) section for details.

