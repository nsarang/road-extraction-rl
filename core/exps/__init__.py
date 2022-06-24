from core.utils.registry import Registry

EXPS = Registry("Policy Experiments")
from .dqn.dqn_v1 import *
from .dqn.dqn_multi import *
from .dqn.seq import *
from .dqn.segrl import *
from .ppo import *
from .rss import *
from .sac import *
from .td3 import *