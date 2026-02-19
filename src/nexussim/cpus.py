from typing import Protocol

from nexussim.observer import Observer


class CPUProvider(Protocol):
    def max_cpus(self) -> float:
        """
        Returns the maximum number of CPUs available.

        This method should be implemented to return the maximum number of CPUs available
        for allocation.

        Returns:
            float: The maximum number of CPUs.
        """

    def cpu_speed(self) -> float:
        """
        Returns the CPU speed.

        This method should be implemented to return the CPU speed in Hz.

        Returns:
            float: The CPU speed.
        """

    def free_cpus(self) -> float:
        """
        Returns the number of CPUs available for allocation.

        This method should be implemented to return the difference between the total
        number of CPUs and the currently allocated CPUs.

        Returns:
            float: The free CPUs.
        """

    def used_cpus(self) -> float:
        """
        Returns the number of CPUs that are currently allocated.

        This method should be implemented to return the sum of all CPU allocations made
        by the request_cpu method, in CPU fractions.

        Returns:
            float: The used CPUs.
        """

    def request_cpu(self, requester: str, cpus_frac: float) -> bool:
        """
        Requests CPUs from the provider.

        This method should be implemented to request a CPU allocation of the given size,
        in CPU fractions.

        Args:
            requester (str): The name of the requester of the CPU allocation. cpus_frac
            (float): The size of the CPU allocation to request, in CPU
                fractions.

        Returns:
            bool: True if the request was successful, False otherwise.
        """

    def release_cpu(self, requester: str) -> bool:
        """
        Releases the CPU resources allocated to a requester.

        This method should be implemented to free the CPU resources previously allocated
        to the requester, making them available for other allocations.

        Args:
            requester (str): The name of the requester whose CPU allocation should be
            released.

        Returns:
            bool: True if the release was successful, False otherwise.
        """

    def flush(self) -> None:
        """
        Flushes the CPU provider, releasing all allocated CPUs.

        This method should be implemented to flush the CPU provider, releasing all
        allocated CPU resources.
        """

    def cpu_usage(self) -> dict[str, float]:
        """
        Returns the CPU usage per requester.
        """


class UnboundedCPU:
    """A CPU that, despite having a limited capacity,
       will not fail ever on CPU requests.

    Created to models where we do not want to deal with lack of CPUs.
    """

    def __init__(
        self,
        max_cpus: float,
        cpu_speed: float = 0.0,
        observers: list[Observer] | None = None,
    ):
        """
        Initializes the UnboundedCPU with a maximum number of CPUs.

        Args:
            max_cpus (float): The maximum number of CPUs that can be allocated.
            cpu_speed (float, optional): The CPU speed in Hz. Defaults to 0.0.
            observers (list[Observer] | None, optional): A list of observers to notify
                when the used CPUs change.

        Initializes the used CPUs to 0.
        """
        self._used_cpus = {}
        self._max_cpus = max_cpus
        self._cpu_speed = cpu_speed
        self._observers = observers

    def max_cpus(self) -> float:
        """
        Returns the maximum number of CPUs available.

        Returns:
            float: The maximum number of CPUs.
        """
        return self._max_cpus

    def cpu_speed(self) -> float:
        """
        Returns the CPU speed.

        Returns:
            float: The CPU speed.
        """
        return self._cpu_speed

    def free_cpus(self) -> float:
        """
        Returns the number of CPUs available for allocation.

        This method returns the difference between the total number of CPUs and the
        currently allocated CPUs.

        Negative numbers means there is more allocated CPU fractions than it is
        available.

        Returns:
            float: The free CPUs.
        """
        return self._max_cpus - self.used_cpus()

    def used_cpus(self) -> float:
        """
        Returns the number of CPUs that are currently allocated.

        This method returns the sum of all CPU allocations at the moment,
        in CPU fractions.

        Returns:
            float: The used CPUs.
        """
        return sum(self._used_cpus.values())

    def request_cpu(self, requester: str, cpus_frac: float) -> bool:
        """
        Requests CPUs from the provider.

        This method requests a CPU allocation of the given size, in CPU fractions.
        The request is never refused, and the requester is always added to the used
        CPUs.

        Args:
            requester (str): The name of the requester of the CPU allocation.
            cpus_frac (float): The size of the CPU allocation to request, in CPU
                fractions.

        Returns:
            bool: Always True.
        """
        self._used_cpus[requester] = cpus_frac
        if self._observers is not None:
            for observer in self._observers:
                observer.update(self)
        return True

    def release_cpu(self, requester: str) -> bool:
        """
        Releases the CPU resources allocated to a requester.

        This method frees the CPU resources previously allocated to the requester,
        making them available for other allocations.

        Args:
            requester (str): The name of the requester whose CPU allocation should be
            released.

        Returns:
            bool: True if the release was successful, False otherwise.
        """
        self.request_cpu(requester, 0)
        if self._observers is not None:
            for observer in self._observers:
                observer.update(self)
        return True

    def flush(self) -> None:
        """
        Flushes the CPU provider, releasing all allocated CPUs.
        """
        self._used_cpus = {}

    def cpu_usage(self) -> dict[str, float]:
        return self._used_cpus.copy()
