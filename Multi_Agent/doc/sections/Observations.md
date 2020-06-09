## Standard Observations and Actions

We have introduced `AgentInterface` in [Agent](Agent.md), which allows us to choose from the standard observation and action types for communication
between an agent and a SMARTS environment.

### Observations

Here we will introduce details of available observation types.
For `AgentType.Full`, which contains the most concrete observation details, the raw observation returned
is a Python `namedtuple` with the following fields,
* `events` a `namedtuple` with the following fields,
    * `collisions` - collisions the vehicle has been involved with other vehicles (if any)
    * `off_road` - `True` if the vehicle is off the road
    * `reached_goal` - `True` if the vehicle has reached its goal
    * `reached_max_episode_steps` - `True` if the vehicle has reached its max episode steps
* `ego_vehicle_state` - a `VehicleState` `namedtuple` for the ego vehicle with the following fields,
    * `heading` - vehicle heading in degrees
    * `speed` - agent speed in km/h
    * `throttle` - a normalized engine force value
    * `brake` - a normalized brake force value
    * `steering` - angle of front wheels in degrees
    * `position` - 3D numpy array (x, y, z) of vehicle position
    * `bounding_box` - `BoundingBox` `namedtuple` for the `width`, `length`, `height` of the vehicle.
* `neighborhood_vehicle_states` - a list of `SocialVehicleState` `namedtuple`s, each with the following fields,
    * `heading`, `speed`, `position`, `bounding_box` - the same as with `ego_vehicle_state`
    * `lane_id` - a globally unique identifier of the lane under this vehicle 
    * `lane_index` - index of the lane under this vehicle, right most lane has index 0 and the index increments to the left
* `top_down_rgb` - A 256x256 RGB image with the ego vehicle at the center
    ![](../assets/rgb.png)
* `occupancy_grid_map` - A 256x256 [OGM](https://en.wikipedia.org/wiki/Occupancy_grid_mapping) around the ego vehicle
* `waypoint_paths` - A list of waypoints in front of the ego vehicle showing the potential routes ahead. Each item is a `Waypoint` instance with the following fields,
    * `id` - an integer identifier for this waypoint
    * `pos` - a numpy array (x, y) center point along the lane
    * `heading` - heading angle of lane at this point (degrees)
    * `lane_width` - width of lane at this point (meters)
    * `speed_limit` - lane speed in km/h
    * `lane_id` - a globally unique identifier of lane under waypoint
    * `right_of_way` - `True` if this waypoint has right of way, `False` otherwise
    * `lane_index` - index of the lane under this waypoint, right most lane has index 0 and the index increments to the left

See implemention in `smarts/core/sensors.py`


Then, you can choose the observations needed through `AgentInterface` and process these raw observations through `observation_adapter`.



### Actions

* `ActionSpaceType.Continuous`: continuous action space with throttle, brake, absolute steering angle. It is a tuple of `throttle` [0, 1], `brake` [0, 1], and `steering` [-1, 1].
* `ActionSpaceType.ActuatorDynamic`: continuous action space with throttle, brake, steering rate. Steering rate means the amount of steering angle change *per second* (either positive or negative) to be applied to the current steering angle. It is also a tuple of `throttle` [0, 1], `brake` [0, 1], and `steering_rate`, where steering rate is in number of degrees per second.
* `ActionSpaceType.Lane`: discrete lane action space of *strings* including "keep_lane",  "slow_down", "change_lane_left", "change_lane_right" as of version 0.3.2b, but a newer version will soon be released. In this newer version, the action space will no longer being strings, but will be a tuple of an integer for `lane_change` and a float for `target_speed`.
