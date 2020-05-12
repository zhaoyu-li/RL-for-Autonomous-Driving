## Scenario Studio

If you have access to sstudio you can use it to generate traffic with different social vehicle numbers and routes, and agent missions:

### generate traffic
```python
traffic_actor = TrafficActor(name="car", speed=Distribution(sigma=0.2, mean=0.8),)

traffic = Traffic(
    flows=[
        Flow(route=RandomRoute(), rate=180, actors={traffic_actor: 1},)  # 3 / minute
        for i in range(10)
    ]
)

gen_traffic(scenario=scenario, traffic=traffic, name="all", seed=seed_, output_dir=output_dir)
```
traffic actor is used as a description/spec for traffic actors (e.x. Vehicles, Pedestrians,
etc). The defaults provided are for a car. You can specify acceleration, deceleration, speed distribution, imperfection 
distribution and other configs for social cars. See more config for `TrafficActor` in `smarts/sstudio/types.py`.

Flow can be used to generate repeated vehicles, you can config vehicle route and depart rate here. 

After run `gen_traffic` function, a dir named "traffic" containing vehicle config xmls will be created under output_dir.
 

The file `scenario/scenario.py` is a short example of how this would

If available it can be generated using the following call:

```bash
  python3 scenario/scenario.py
```

### generate missions
SMARTS allows generate missions for ego agents and social agents, which is similar to routes for social vehicles.
Also, "missions.rou.xml" file will be created under output dir.
```python
# agent missions
gen_missions(
    scenario,
    missions=[Mission(Route(begin=("edge0", 0, "random"), end=("edge1", 0, "max"),)),],
    seed=seed_,
)
```

### create new maps
To enrich your training datasets, you can edit your own map through [sumo NETEDIT](https://sumo.dlr.de/docs/NETEDIT.html) and export it in a map.net.xml format.
And if you have an additional file you wish to turn into a map you can use the conversion utilities like:

```bash
  python3 smarts/sstudio/sumo2mesh.py dataset_public/2lane_sharp/map.net.xml dataset_public/2lane_sharp/map.glb --format=glb
  python3 smarts/sstudio/sumo2mesh.py dataset_public/2lane_sharp/map.net.xml dataset_public/2lane_sharp/map.egg --format=egg
```
