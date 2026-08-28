from enum import Enum

class Stage(str, Enum):
    DISCOVER = "DISCOVER"
    HYPOTHESIZE = "HYPOTHESIZE"
    BUILD = "BUILD"
    BACKTEST = "BACKTEST"
    ATTACK = "ATTACK"
    VALIDATE = "VALIDATE"
    RANK = "RANK"
    PINE = "PINE"

PIPELINE = [
    Stage.DISCOVER, Stage.HYPOTHESIZE, Stage.BUILD, Stage.BACKTEST,
    Stage.ATTACK, Stage.VALIDATE, Stage.RANK, Stage.PINE,
]
