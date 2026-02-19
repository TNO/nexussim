from nexussim.context import Context
from nexussim.cpus import CPUProvider
from nexussim.memory import MemoryProvider
from nexussim.network import NetworkProvider
from nexussim.segments.base import Segment


class Compute:
    def __init__(
        self,
        memory: MemoryProvider,
        cpu: CPUProvider,
        networks: dict[str, NetworkProvider] = None,
        description: str = "",
    ):
        """
        Initializes a Compute object with the given memory and CPU providers, and
        optional parameters for the CPU speed and a description.

        Args:
            memory (MemoryProvider): The memory provider for the compute object.
            cpu (CPUProvider): The CPU provider for the compute object.
            networks (dict, optional): Dictionary of network providers attached to this compute object.
            description (str, optional): A description for the compute object. Defaults
                to an empty string.
        """
        self._memory = memory
        self._cpu = cpu
        self._networks = networks if networks is not None else {}
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

    def network(self, name: str) -> NetworkProvider:
        """
        Gets a network by name.

        Args:
            name (str): The name of the network.

        Returns:

            NetworkProvider or None:
                The network provider if the network exists under the given name,
                otherwise None.
        """
        return self._networks.get(name)

    @property
    def networks(self):
        """
        Gets all the network names.

        Returns:
            list: A list of all network names tied to this compute.
        """
        return list(self._networks)

    @property
    def memory(self):
        return self._memory

    def execute(self, segment: Segment):
        segment.execute(self.context)
