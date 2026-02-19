class Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.__storage = {}
        # makes sure 'compute' exists in the context
        if "compute" not in self.__dict__:
            self.compute = None

    def store(self, **kwargs):
        self.__storage.update(kwargs)

    def retrieve(self, key):
        return self.__storage.get(key)
