# this file is for generating social agents and their missions

from pathlib import Path

from smarts.sstudio import gen_traffic
from smarts.sstudio.types import (
    Traffic,
    Flow,
    RandomRoute,
    TrafficActor,
    Distribution,
)

scenario_dir = (Path(__file__).parent / "f1_public").resolve()
scenario_names = ["shanghai", "silverstone", "monte", "interlagos"]
scenario_paths = [scenario_dir / name for name in scenario_names]
print(scenario_paths)

sv_nums = [60, 60, 60, 60]

seed = 45

traffic_actor = TrafficActor(name="car", speed=Distribution(sigma=0.1, mean=0.3),)

for scenario_path, sv_num in zip(scenario_paths, sv_nums):
    traffic = Traffic(
        flows=[
            Flow(
                route=RandomRoute(),
                begin=0,
                end=1
                * 60
                * 60,  # make sure end time is larger than the time of one episode
                rate=60,
                actors={traffic_actor: 1},
            )
            for i in range(sv_num)
        ]
    )

    print(f"generate flow with {sv_num} social vehicles in {scenario_path.name} ")

    gen_traffic(
        scenario_path,
        traffic,
        name="all",
        output_dir=scenario_path,
        seed=seed,
        overwrite=True,
    )
