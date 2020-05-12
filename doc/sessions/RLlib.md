## RLlib
**RLlib** is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety 
of applications. RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic.

### Recommend Read First
There are many docs about ray and rllib, to get started with the competition example. We recommend to read the following
pages first.

- [RLlib in 60 seconds](https://docs.ray.io/en/latest/rllib.html#rllib-in-60-seconds): get started with RLlib
- [Common Parameters](https://docs.ray.io/en/latest/rllib-training.html#common-parameters): see common tune configs.
- [Basic Python API](https://docs.ray.io/en/latest/rllib-training.html#basic-python-api): see basic `tune.run` function.
- [Callbacks and Custom Metrics](https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics): see callbacks and metrics.
- [Visualizing Custom Metrics](https://docs.ray.io/en/latest/rllib-training.html#visualizing-custom-metrics): use tensorboard to visualize metrics.
- [Built-in Models and Preprocessors](https://docs.ray.io/en/latest/rllib-models.html#default-behaviours): see built-in preprocessor how to deal with different observation space.
- [Proximal Policy Optimization (PPO)](https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo): see PPO and PPO parameters.
- [PopulationBasedTraining](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#populationbasedtraining): see Population Based Training algorithm and examples. 
- [RLlib Examples](https://docs.ray.io/en/latest/rllib-examples.html): see RLlib examples to get known about it quickly.

### SMARTS RLlib Tips

#### resume or continue training
If you want to continue an aborted experiemnt. you can set `resume=True` in `tune.run`. But note that`resume=True` will continue to use the same configuration as was set in the original experiment.
To make changes to a started experimented, you can edit the latest experiment file in `~/ray_results/rllib_example`.

Or if you want to start a new experiment but train from an exist checkpoint, you can set `restore=checkpoint_path` in `tune.run`.

