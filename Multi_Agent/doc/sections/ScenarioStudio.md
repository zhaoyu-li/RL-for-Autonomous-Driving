## Scenario Studio

The Scenario Studio of SMARTS is a stand alone package (`sstudio`) that supports flexible and expressive scenario specification.
If you have access to `sstudio` you can use it to generate traffic with different traffic vehicle numbers and routes, and agent missions.

### Generate traffic
```python
traffic_actor = TrafficActor(name="car", speed=Distribution(sigma=0.2, mean=0.8),)

# add 10 social vehicles with random routes.
traffic = Traffic(
    flows=[
        # generate flows last for 10 hours
        Flow(route=RandomRoute(), begin=0, end=10 * 60 * 60, rate=25, actors={traffic_actor: 1},)
        for i in range(10)
    ]
)

gen_traffic(scenario_path, traffic, name="all", output_dir=scenario_path, seed=seed_, overwrite=True)
```
`traffic_actor` is used as a spec for traffic actors (e.g. Vehicles, Pedestrians, etc). The defaults provided are for a car.
You can specify acceleration, deceleration, speed distribution, imperfection distribution and other configs for social cars.
See more config for `TrafficActor` in `smarts/sstudio/types.py`.

Flow can be used to generate repeated vehicle runs on the same route, you can config vehicle route and depart rate here. 

After the `gen_traffic` function is run, a dir named "traffic" containing vehicle config xmls will be created under output_dir.
 

The file `scenario/scenario.py` is a short example of how this all works.

We just run a `scenario.py` file as a regular Python script to generate scenarios:

```bash
  python3 scenario/scenario.py
```

IMPORTANT: if you want to train model on one scenario, remember to set the `end` time of flow larger or equal to your expected
training time, since the smarts will continue the flow after each `reset` called. However, if there are multi scenarios to train
for one worker, you can relax this restriction since after the scenario change, the flow will also be reset to the beginning time.

### Generate missions
The Scenario Studio of SMARTS also allows generation of *missions* for ego agents and social agents. These missions are similar
to routes for social vehicles. When we run `gen_missions`, "missions.rou.xml" file will be created under the output dir:
```python
# agent missions
gen_missions(
    scenario,
    missions=[Mission(Route(begin=("edge0", 0, "random"), end=("edge1", 0, "max"),)),],
    seed=seed_,
)
```

### Create new maps
To enrich your training datasets, you can edit your own map through [SUMO's NETEDIT](https://sumo.dlr.de/docs/NETEDIT.html) and export it in a map.net.xml format.
And if you have an additional file you wish to turn into a map you can use the conversion utility `sumo2mesh.py` like this:

```bash
  python3 -m smarts.sstudio.sumo2mesh dataset_public/2lane_sharp/map.net.xml dataset_public/2lane_sharp/map.glb --format=glb
  python3 -m smarts.sstudio.sumo2mesh dataset_public/2lane_sharp/map.net.xml dataset_public/2lane_sharp/map.egg --format=egg
```
