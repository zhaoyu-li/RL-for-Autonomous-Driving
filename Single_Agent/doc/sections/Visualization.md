## Visualization

### Envision

SMARTS includes a visualizer called *Envision* that runs on a process separate from the main simulation process. To manage these processes we use *supervisord* (available as a *pip* dependency). Supervisord knows what processes to run by reading a `supervisord.conf` file which is included with your Starter Kit. Instead of running `python random_example/run.py` directly, simply call,

```bash
supervisord
```

The `supervisord.conf` file contains,

```bash
# [program:smarts]
# command=python random_example/run.py
# ...
# [program:envision_server]
# command=envision_server --scenarios ./dataset_public --port 8081
# ...
```

Feel free to change the above commands as necessary (the first to specify the command to run for SMARTS, and the latter to adjust location of the scenarios to use so that Envision can load them).

To see the front-end visualization visit `http://localhost:2310/` in your browser. Select the simulator instance in the top left dropdown. If you are using SMARTS on a remote machine you will need to port forward both 2310 and 8081.

### Visdom

We also have built-in support for [Visdom](https://github.com/facebookresearch/visdom) so you can see the image-based observation outputs in real-time. Start the visdom server before running your scenario and open the server URL in your browser `http://localhost:8097`.

```bash
# (optional) source your virtual environment
cd ~/src/starter_kit
source .venv/bin/activate

# install visdom
pip install visdom

# start the server
visdom
```

In your `run.py` script, enable `visdom` with,

```python
env = gym.make(
        "smarts.env:hiway-v0", # env entry name
        ...
        visdom=True, # whether or not to enable visdom visualization (see Appendix).
        ...
    )
```
