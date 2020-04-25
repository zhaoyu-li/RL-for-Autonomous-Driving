import argparse
from pathlib import Path

from smarts.core import seed
from smarts.sstudio import gen_traffic
from smarts.sstudio.types import (
    Traffic,
    Flow,
    RandomRoute,
    TrafficActor,
)


def generate_scenario(scenario_root, n_sv, output_dir):
    name = "random_" + str(n_sv)
    traffic = Traffic(
        flows=[
            Flow(
                route=RandomRoute(),
                rate=1,
                begin=0,
                end=1,
                actors={TrafficActor(name="car"): 1.0},
            )
            for n in range(n_sv)
        ]
    )

    gen_traffic(scenario_root, traffic, name=name, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate social vehicle for scenario")

    # env setting
    parser.add_argument(
        "--scenario", type=str, default=None, help="Path to the scenario"
    )
    parser.add_argument(
        "--num_social_vehicles", type=int, default=10, help="Number of social vehicles"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for social vehicle traffic files",
    )
    args = parser.parse_args()

    generate_scenario(str(Path(args.scenario).absolute()), args.nv, args.output_dir)
