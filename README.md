# METROID II: RETURN OF SAMUS AI
<p align="center">
    <img alt="Metroid II Logo" src="/assets/logo.png" height="200">
</p>

__This is a reinforcement learning artificial intelligence model built to play the GameBoy game, Metroid II: Return of Samus.__

## Checkout the YouTube video about the AI and the model breakdown
<img alt="YouTube thumbnail" src="/assets/thumbnail.png" height="200"> <img alt="YouTube breakdown thumbnail" src="/assets/tb_thumbnail.png" height="200">


## 👾 Getting Started 👾
1. Clone this repo
2. Install correct python version: 3.10.13
    * if using anaconda, you can create a new environment by running these commands in terminal
    ```
    conda create --name metroidai python=3.10.13
    conda activate metroidai
    ```
    * you can also use pyenv, some other python environment handler, or just insall python 3.10.13 to your machine
    * This probably will work with other versions of python, but it hasn't been tested
3. Open up the repo in terminal and run
```
pip install -r requirements.txt
```
4. Create a folder called ```ROMs``` in the root of the repository
```
mkdir ROMs
```
5. Legally obtain a Metroid II ROM and copy the .gb file into the ```ROMs/``` directory
    * You also have to rename the file to ```Metroid2.gb```

## 🤖 Run Pretrained Model 🤖
1. Enter ```src/``` directory
```
cd src
```
2. Run ```run_pretrained_model.py``` file
```
python run_pretrained_model.py
```
3. To stop the run, press ```CTRL + C``` in the terminal

## 🦾 Train Your Own AI 🦾

1. Enter ```src/``` directory
```
cd src
```
2. Run ```train.py``` file
```
python train.py
```

### Tips For Training
Unless you have a very powerful computer, and A LOT (and I mean A LOT) of time, I would recommend the following changes:
* decrease the ```max_iter``` field in the configuration you're using to reduce time
* decrease ```n_envs``` field in the configuration you're using to reduce cpu usage
* decrease ```n_epochs``` in ```train.py``` to reduce time
* decrease ```learning_iters``` in ```train.py``` to reduce time
* decrease ```batch_size``` argument in the ```PPO``` model in ```train.py``` to decrease memory load
* decrease ```n_steps``` argument in the ```PPO``` model in ```train.py``` to decrease memory load

## 🔨 Troubleshooting 🔨
If you have issues running the model for both the pretrained and/or training files, try these steps:
* Make sure you are running ```train.py``` or ```run_pretrained_model.py``` from ```src/``` directory
* Set ```n_envs``` to ```1``` in the configuration you're using if you have issues with parallelizing the model
* Make sure correct versions of libraries in requirements.txt are being used
* Make sure ffmpeg is installed
* Make sure Python 3.10.13 is used

## 💡 Built With 💡
This AI couldn't have been done without these amazing projects. Please check them out and support them!

### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
    <img alt="PyBoy Logo" src="/assets/pyboy-logo.png" height="100">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
    <img alt="Stable Baselines 3 Logo" src="/assets/stable-baselines-logo.png" height="100">
</a>

### [Return Of Samus Disassembly](https://github.com/alex-west/M2RoS)
<a href="https://github.com/alex-west/M2RoS">
    <img alt="Metroid II Cartridge" src="/assets/m2-cartridge.jpeg" height="100">
</a>

# Thanks for visiting! Continue the discussion in these awesome communities:
[![Discord Banner 2 PyBoy](http://invidget.switchblade.xyz/bEMadYckBS)](https://discord.gg/bEMadYckBS) 
[![Discord Banner 2 Metconst](http://invidget.switchblade.xyz/XnfmbNcSjr)](https://discord.gg/XnfmbNcSjr)
[![Discord Banner 2 Reinforcement Learning](http://invidget.switchblade.xyz/pV8k2v6Fes)](https://discord.gg/pV8k2v6Fes)


<p align="center">
    <img alt="Metroid II Box Art" src="/assets/boxart.jpg" height="200" >
</p>