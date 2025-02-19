from .shell_game_push import ShellGamePushEnv
from .shell_game_pick import ShellGamePickEnv
from .shell_game_touch import ShellGameTouchEnv
from .intercept import InterceptSlowEnv, InterceptMediumEnv, InterceptFastEnv
from .intercept_grab import InterceptGrabSlowEnv, InterceptGrabMediumEnv, InterceptGrabFastEnv
from .rotate_lenient import RotateLenientEnvPos, RotateLenientEnvPosNeg
from .rotate_strict import RotateStrictEnvPos, RotateStrictEnvPosNeg
from .take_it_back import TakeItBackEnv
from .remember_color import RememberColor3Env, RememberColor5Env, RememberColor9Env
from .remember_shape import RememberShape3Env, RememberShape6Env, RememberShape9Env
from .remember_shape_and_color import RememberShapeAndColor3x2Env, RememberShapeAndColor3x3Env, RememberShapeAndColor5x3Env
from .bunch_of_colors import BunchOfColors3Env, BunchOfColors5Env, BunchOfColors7Env
from .seq_of_colors import SeqOfColors3Env, SeqOfColors5Env, SeqOfColors7Env
from .chain_of_colors import ChainOfColors3Env, ChainOfColors5Env, ChainOfColors7Env