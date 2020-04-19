# Further reading

If you have not read it yet read [README.md](README.md) before continuing

# random & rllib examples

`random_example/run.py` can be run easily. It is just intended for testing.

```bash
    python3 ~/src/starter_kit/random_example/run.py
```

Likewise `rllib_example/run.py`.

```bash
    python3 ~/src/starter_kit/rllib_example/run.py
```

# The Agent

Agent initialization takes the form:

```python
    agent = Agent(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        policy=Policy(),
        observation_space=OBSERVATION_SPACE,
        action_space=ACTION_SPACE,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    )
```

The following sections explain what each of the parametres mean.

If you wish to look at an example of an agent please look at `starter_kit/rllib_example/agent.py`. In that file you will find much of what is explained here put to use.

## AgentInterface

There are a few ways to create the agent interface but the suggested way is to use a preset:

```python
    agent_interface = AgentInterface.from_type(
        interface = AgentType.Standard,
        max_episode_steps = 100, 
        ...
    )
```

**With the following interface presets please note the observations and action spaces they have available.**

The agent interface you will use unless otherwise specified: 
 - `AgentType.Standard`  
    - Waypoints
    - Neighborhood vehicles
    - Continuous action space

A powerful but expensive agent interface: 
 - `AgentType.Full`  
    - Waypoints
    - Neighborhood vehicles
    - Continuous action space
    - RGB
    - OGM
    - LIDAR

Useful if you are just testing out a map but otherwise should not be used:
 - `AgentType.Laner`
    - Waypoints
    - Lane action space

It is not recommended to use any of the other preset options.


## Policy

A policy is a provider that takes in the observations of an agent and decides on an action.

```python
    # A simple policy that ignores observations
    class IgnoreObservationsPolicy(AgentPolicy):
        def act(self, obs):
            return [throttle, brake, steering]
```
The observation passed in should be the observations that a given agent sees. In **contininuous action space** the policy is expected to pass out values for `throttle` [0->1], `brake` [0->1], and `steering` [-1->1]. 

Otherwise, only while using **lane action space**, the policy is expected to return a laning related command: `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, `"change_lane_right"`

## Adapters and Spaces

Adapters convert the data such as an agent's raw sensor observations to a more useful form. And spaces provide samples for variation. Adapters and spaces are particularly relevant to the `rllib.py` example.

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

## Agent Observations

Of all the information to work with it is useful to know a bit about the main agent observations in particular.

For that see the `## Observation Space` section in README.md

## sstudio

If you have access to sstudio you can use it to generate traffic, social agents, and agent missions:

The file `dataset_public/2lane_sharp_bwd/scenario.py` is a short example of how this would

If available it can be generated using the following call:

```bash
  python3 dataset_public/2lane_sharp_bwd/scenario.py
```

And if you have an additional file you wish to turn into a map you can use the conversion utilities like:

```bash
  python3 smarts/sstudio/sumo2mesh.py dataset_public/2lane_sharp_bwd/map.net.xml dataset_public/2lane_sharp_bwd/map.glb --format=glb
  python3 smarts/sstudio/sumo2mesh.py dataset_public/2lane_sharp_bwd/map.net.xml dataset_public/2lane_sharp_bwd/map.egg --format=egg
```