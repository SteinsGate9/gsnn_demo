class AttrProxy(object):
    """
    Just showing off
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix
    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))