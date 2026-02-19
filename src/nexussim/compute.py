from nexussim.context import Context
from nexussim.cpus import CPUProvider
from nexussim.memory import MemoryProvider
from nexussim.segments.base import Segment


class Compute:
    def __init__(self, memory: MemoryProvider, cpu: CPUProvider):
        self.memory = memory
        self.cpu = cpu
        self._context = Context(compute=self)

    @property
    def context(self):
        return self._context

    def execute(self, segment: Segment):
        segment.execute(self.context)
