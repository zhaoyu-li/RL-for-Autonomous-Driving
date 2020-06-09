# Competition Stage 2
- `agent.py`: 
    - add different load model approach for different action space; 
    - add keep_lane_agent for test.
    - increase max step due to fit the large map.
- `custom_observations.py`: 
    - do normalizing to steering, speed; 
    - change default ttc from 1000 to 1(similar intention for normalize); 
    - extended `_ego_ttc_calc` to support 5 lanes.
- `scenario.py`:
    - add more vehicles and increase flow rate due to the large map;
- `run.py`:
    - no difference, can use keep lane agent for test.
- `trainer.py`:
    - add two training example to train on multi maps. 
    

