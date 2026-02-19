from nexussim.context import Context
from nexussim.cpus import CPUProvider
from nexussim.memory import MemoryProvider
from nexussim.segments.base import Segment


class Compute:
    def __init__(
        self,
        memory: MemoryProvider,
        cpu: CPUProvider,
        description: str = "",
    ):
        """
        Initializes a Compute object with the given memory and CPU providers, and
        optional parameters for the CPU speed and a description.

        Args:
            memory (MemoryProvider): The memory provider for the compute object.
            cpu (CPUProvider): The CPU provider for the compute object.
            cpu_speed (float, optional): The CPU speed for the compute object in Hz.
                Defaults to 1e9.
            description (str, optional): A description for the compute object. Defaults
                to an empty string.
        """
        self.memory = memory
        self._cpu = cpu
        self._desc = description
        self._context = Context(compute=self)

    @property
    def description(self):
        return self._desc

    @property
    def context(self):
        return self._context

    @property
    def cpu(self):
        return self._cpu

    def execute(self, segment: Segment):
        segment.execute(self.context)
