from gym import spaces
from env.multi_discrete import MultiDiscrete


def get_shape(input_space):
    if (isinstance(input_space, spaces.Box)):
        if (len(input_space.shape) == 1):
            return input_space.shape[0]
        else:
            return input_space.shape
    elif (isinstance(input_space, spaces.Discrete)):
        return input_space.n
    elif (isinstance(input_space, MultiDiscrete)):
        return sum(input_space.high - input_space.low + 1)
    else:
        print('[Error] shape is {}, not Box or Discrete or MultiDiscrete'.
              format(input_space.shape))
        raise NotImplementedError
