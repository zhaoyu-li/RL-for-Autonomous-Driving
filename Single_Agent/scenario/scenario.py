import os
import random

from smarts.core import seed
from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Mission,
    Distribution,
    LaneChangingModel,
    JunctionModel,
)

seed_ = 3
seed(seed_)

scenario_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../dataset_public/3lane"
)

traffic_actor = TrafficActor(name="car", speed=Distribution(sigma=0.2, mean=0.8),)


# add 10 social vehicles with random routes.
traffic = Traffic(
    flows=[
        # generate flows last for 10 hours
        Flow(
            route=RandomRoute(),
            begin=0,
            end=10 * 60 * 60,
            rate=25,
            actors={traffic_actor: 1},
        )
        for i in range(10)
    ]
)

gen_traffic(
    scenario_path,
    traffic,
    name="all",
    output_dir=scenario_path,
    seed=seed_,
    overwrite=True,
)
