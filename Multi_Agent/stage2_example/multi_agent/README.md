# Competition Stage 2
- `agent.py`: contents same with single agent
- `agent_prefabs`: register social agent
- `custom_observations.py`: contents same with single agent
- `scenario.py`:
    - generate social agent missions
    - generate agent missions (to make them start from the same position line)
    - generate social vehicles (contents same with single agent)
- `run.py`: contents same with single agent(but will auto load social agent and agent missions and run with social agents)
- `trainer.py`: contents same with single agent(but will auto load social agent and agent missions and run with social agents)
    

# Suggest step:
copy `f1_public` to `multi_agent`
- run `scenario.py` to generate social agent missions, agent missions, social vehicles
- run `trainer.py` to train agents
- run `run.py` to evaluate agents
