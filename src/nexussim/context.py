class Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # makes sure 'compute' exists in the context
        if "compute" not in self.__dict__:
            self.compute = None
