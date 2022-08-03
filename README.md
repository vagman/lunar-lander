### Lanar Lander (Intro in Reinforcement Learning)
A project for a University assignment in Machine Learning Agents class, in the 8th semester, using Python.

### Setup & Run
1. ```$ pip install gym[box2d]```
2. ```$ pip install stable-baselines3```
3. Run as ```$ python3 reinforcement-learning.py```. As soon as a couple of models are trained you can either Ctrl + C the program or wait for it to reach the timesteps goal. 
4. Edit ```line 4``` of ```evaluate.py``` with the model you wish to evaluate e.g. ```model_path = f"{models_dir}/2600000.zip"```. Then, run as ```$ python3 evaluate.py```.

### General Info
* [Python](https://www.python.org/) 3.10.5
* [gym](https://pypi.org/project/gym/) 0.21.0
* [stable-baselines3](https://stable-baselines3.readthedocs.io/) 1.6.0
