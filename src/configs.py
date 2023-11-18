# ALL custom configs must have the same fields
basic = {
    "action_frequency": 5,
    "states": ["../states/bottom_of_pit.state",
               "../states/inside_pit.state",
               "../states/past_first_door.state",
               "../states/post_start_screen.state"],
    "rom_path": "../ROMs/Metroid2.gb",
    "seed": None,
    "max_steps": 32768,
    "window": 'headless',
    "n_envs": 6
}

short = {
    "action_frequency": 5,
    "states": ["../states/bottom_of_pit.state",
               "../states/inside_pit.state",
               "../states/past_first_door.state",
               "../states/post_start_screen.state"],
    "rom_path": "../ROMs/Metroid2.gb",
    "seed": None,
    "max_steps": 500,
    "window": 'SDL2',
    "n_envs": 1
}

replay = {
    "action_frequency": 5,
    "states": ["../states/past_first_door.state"],
    "rom_path": "../ROMs/Metroid2.gb",
    "seed": None,
    "max_steps": 5000,
    "window": 'SDL2',
    "n_envs": 1
}
