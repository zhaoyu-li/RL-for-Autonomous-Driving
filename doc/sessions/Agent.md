## Agent

SMARTS provide users the ability to custom their agents. `Agent` class have below fields, see more details in 
`smarts/env/agent.py` 

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

An `Agent` class instance is shown below.
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

We will further explain the fields of `Agent` class
### AgentInterface

`AgentInterface` is used to wrap and enable agent observation and action flow. To create an agent interface, you can try
```python
agent_interface = AgentInterface.from_type(
    interface = AgentType.Standard,
    max_episode_steps = 1000, 
    ...
)
```

SMARTS provide some interface types, and the differences between them is shown in below table. **T** means the `AgentType` will provide this option or information. 

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

max_episode_steps controls agent max running step in an episode, None default means agents have no max_episode_step limit.
You can move max_episode_steps control authority to RLlib with config option `horizon`, but lose the ability to custom 
different max_episode_len for each agent.

action controls the agent action type used. There are three `ActionSpaceType`: ActionSpaceType.Continuous, ActionSpaceType.Lane 
and ActionSpaceType.ActuatorDynamic.
- `ActionSpaceType.Continuous`: continuous action space with throttle, brake, steering.
- `ActionSpaceType.ActuatorDynamic`: continuous action space with throttle, brake, additional_steering. Additional steering means
the final steer action will be a weighted sum between last_steering_angle and additional_steering 
- `ActionSpaceType.Lane`: discrete lane action space including "keep_lane",  "slow_down", "change_lane_left", "change_lane_right".

for other observation options, see [Space](Space.md) for details.

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


## Policy

A policy is a provider that takes in the observations of an agent and decides on an action.

```python
# A simple policy that ignores observations
class IgnoreObservationsPolicy(AgentPolicy):
    def act(self, obs):
        return [throttle, brake, steering_rate]
```
The observation passed in should be the observations that a given agent sees. In **contininuous action space** the policy is expected to pass out values for `throttle` [0->1], `brake` [0->1], and `steering_rate` [-1->1].

Otherwise, only while using **lane action space**, the policy is expected to return a laning related command: `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, `"change_lane_right"`

The policy is needed when we used Agent for evaluation. Otherwise it is not necessary for RL training.

## Adapters and Spaces

Adapters convert the data such as an agent's raw sensor observations to a more useful form. And spaces provide samples for variation.
 Adapters and spaces are particularly relevant to the `rllib_example/agent.py` example. Also check `smarts/env/custom_observations` for some processing examples.

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

The reward adapter does not have a space but allows further processing on the reward if desired.

```python
    def reward_adapter(env_obs, env_reward):
        return env_reward
```

Similarly, the info adapter allows further processing on the info.
```python
def info_adapter(env_obs, env_reward, env_info):
    env_info[INFO_EXTRA_KEY] = "blah"
    return env_info
```

## Agent Observations

Of all the information to work with it is useful to know a bit about the main agent observations in particular.

For that see the [Space](Space.md) section for details.

