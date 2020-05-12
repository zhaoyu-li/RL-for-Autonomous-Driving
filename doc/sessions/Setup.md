## Setup

To setup the simulator (called SMARTS) run the following commands,

```bash
# unzip the starter_kit and place somewhere convenient on your machine. (e.x. ~/src/starter_kit)

cd ~/src/starter_kit
./install_deps.sh
# ...and, follow any on-screen instructions

# test that the sumo installation worked
sumo-gui

# setup virtual environment (Python 3.7 is required)
# or you can use conda environment if you like.
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install the dependencies
pip install smarts-0.3.2b-py3-none-any.whl
pip install smarts-0.3.2b-py3-none-any.whl[train]

# download the public datasets from Codalab to ./dataset_public

# test that the sim works
python random_example/run.py
```


### Common Questions
1. Exception: Could not open window.

This is may be due to some old dependencies of panda3D. Try following instructions to solve it.
```bash
# set DISPLAY 
vim ~/.bashrc
export DISPLAY=":1"
source ~/.bashrc

# set xorg server
sudo wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf
sudo /usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY &
```

