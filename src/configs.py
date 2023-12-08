# ALL custom configs must have the same fields
basic = {
    "action_frequency": 5,
    "states": ["../states/chkpt_1.state",
               "../states/chkpt_2.state",
               "../states/chkpt_3.state",
               "../states/chkpt_4.state",
               "../states/chkpt_5.state",
               "../states/chkpt_6.state",
               "../states/chkpt_7.state",
               "../states/chkpt_8.state",
               "../states/chkpt_9.state",
               "../states/chkpt_10.state",
               "../states/chkpt_11.state"],
    "rom_path": "../ROMs/Metroid2.gb",
    "seed": None,
    "max_steps": 32768,
    "window": 'headless',
    "n_envs": 10,
    "save_rewards": True
}

short = {
    "action_frequency": 5,
    "states": ["../states/chkpt_1.state",
               "../states/chkpt_2.state",
               "../states/chkpt_3.state",
               "../states/chkpt_4.state",
               "../states/chkpt_5.state",
               "../states/chkpt_6.state",
               "../states/chkpt_7.state",
               "../states/chkpt_8.state",
               "../states/chkpt_9.state",
               "../states/chkpt_10.state",
               "../states/chkpt_11.state"],
    "rom_path": "../ROMs/Metroid2.gb",
    "seed": None,
    "max_steps": 512,
    "window": 'headless',
    "n_envs": 10,
    "save_rewards": False
}

replay = {
    "action_frequency": 5,
    "states": ["../states/chkpt_1.state"],
    "rom_path": "../ROMs/Metroid2.gb",
    "seed": None,
    "max_steps": 5000,
    "window": 'SDL2',
    "n_envs": 1,
    "save_rewards": False
}
