from functools import partial

from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
from smac.env import MultiAgentEnv, StarCraft2Env
from .matrix_game.matrix_game_simple import Matrixgame

# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["matrix_game"] = partial(env_fn, env=Matrixgame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
