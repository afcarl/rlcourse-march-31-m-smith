"""
This module implements the base environment class which environments extend.
"""

class Environment():
    """ The environment class
    """
    def __init__(self):
        self.rwd = lambda x: 0
        self.feat = lambda x: 0
        self.act = lambda x: 0
        self.trans = lambda x, y: 1 # returns a distribution over future states

    def attr_getter_setter(self, attr, *args):
        try:
            setattr(self, attr, args[0])
            return self
        except IndexError:
            return getattr(self, attr)

    def reward(self, *args):
        return self.attr_getter_setter("rwd", *args)

    def features(self, *args):
        return self.attr_getter_setter("feat", *args)

    def actions(self, *args):
        return self.attr_getter_setter("act", *args)

    def transition(self, *args):
        return self.attr_getter_setter("trans", *args)

if __name__ == '__main__':
    env = Environment()
    print(env.actions(lambda x: x).actions()(5))


