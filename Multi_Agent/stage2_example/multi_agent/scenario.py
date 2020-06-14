# this file is for generating social agents and their missions

import os
from pathlib import Path

from smarts.sstudio import gen_traffic, gen_missions, gen_social_agent_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    SocialAgentActor,
    EndlessMission,
    RandomRoute,
    TrafficActor,
    Distribution,
)

# Scenario Paths
scenario_dir = (Path(__file__).parent / "f1_public").resolve()
scenario_names = ["shanghai", "silverstone", "monte", "interlagos"]
scenario_paths = [scenario_dir / name for name in scenario_names]

# select it by sumo NETEDIT
starting_edges = ["-156328679", "gneE9", "179968249", "138566998"]


#########################################
# generate social agents with predefined policy
#########################################
#
# N.B. You need to have the agent_locator in a location where the left side can be resolved
#   as a module in form:
#       "this.resolved.module:attribute"
#   In your own project you would place the prefabs script where python can reach it

social_agent1 = SocialAgentActor(
    name="zoo-car1", agent_locator="agent_prefabs:zoo-agent1-v0",
)

social_agent2 = SocialAgentActor(
    name="zoo-car2", agent_locator="agent_prefabs:zoo-agent2-v0",
)

social_agent3 = SocialAgentActor(
    name="zoo-car3", agent_locator="agent_prefabs:zoo-agent2-v0",
)

# here define social agent type and numbers
social_agents = [social_agent2, social_agent3, social_agent1]

# generate social agent missisons and store social agents' information under social_agents/
for scenario, starting_edge in zip(scenario_paths, starting_edges):
    for i, social_agent in enumerate(social_agents):
        gen_social_agent_missions(
            scenario,
            social_agent_actor=social_agent,
            name=f"s-agent-{social_agent.name}",
            missions=[
                # edge_id, lane_index, offset
                EndlessMission(begin=(starting_edge, i + 1, 0),),
                # Mission(Route(begin=("edge-east", 1, 0), end=("edge-east", 1, -5))),
            ],
            overwrite=True,
        )

print("generate social agent missions finished")


#########################################
# generate agent missions
#########################################

# generate agent Missions so agents will born in the same position after env reset()
# Otherwise, no defined agent mission will lead to agent born in different place after env reset()
for scenario, starting_edge in zip(scenario_paths, starting_edges):
    gen_missions(
        scenario=scenario,
        missions=[
            # edge_id, lane_index, offset
            EndlessMission(begin=(starting_edge, 0, 0),),
            # Mission(Route(begin=("edge-east", 0, 0), end=("edge-east", 0, -5))),
        ],
        overwrite=True,
    )

print("generate ego agent missions finished")


#########################################
# generate social vehicles
#########################################
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

print("generate social vehicle flows finished")
