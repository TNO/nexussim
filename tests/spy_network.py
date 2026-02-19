class SpyNetwork:
    def __init__(self, base):
        self._base = base
        self.sent = []

    def __getattr__(self, name):
        return getattr(self._base, name)

    def send(self, from_, to_, data_descriptor):
        self.sent.append((from_, to_, data_descriptor))
        return self._base.send(from_, to_, data_descriptor)
