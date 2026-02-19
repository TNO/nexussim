from typing import Protocol

from pydynaa import DynAASim, Entity, EventExpression, EventHandler, EventType

from nexussim.observer import Observer

CPU_CYCLES_COMPLETED = EventType("CPU_CYCLES_COMPLETED", "The CPU completed a given amount of cycles")
CPU_USAGE_CHANGE = EventType("CPU_USAGE_CHANGED", "The usage of the CPU has updated")


class CPUProvider(Protocol):
    """Protocol describing a CPU provider implementation."""

    def max_cpus(self) -> float:
        """
        Returns:
            float: The maximum number of CPUs available for allocation.
        """
        raise NotImplementedError

    def cpu_speed(self) -> float:
        """
        Returns:
            float: The CPU speed in Hz.
        """

    def free_cpus(self) -> float:
        """
        Returns:
            float: The difference between the total number of CPUs
                   and the currently allocated CPUs (the free CPUs).
        """
        raise NotImplementedError

    def used_cpus(self) -> float:
        """
        Returns:
            float: The sum of all CPU allocations made by the request_cpu method,
                   in CPU fractions (the used CPUs).
        """
        raise NotImplementedError

    def request_cpu(self, requester: str, cpus_frac: float, cpu_cycles: int) -> EventExpression:
        """
        Requests a CPU allocation of the given size from the provider, in CPU fractions.

        Args:
            requester (str): The name of the requester of the CPU allocation.
            cpus_frac (float): Size of the CPU allocation to request in CPU fractions.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        raise NotImplementedError

    def release_cpu(self, requester: str) -> bool:
        """
        Releases all CPU resources previously allocated to a requester,
        making them available for other allocations.

        Args:
            requester (str): The name of the requester
                             whose CPU allocation should be released.

        Returns:
            bool: True if the release was successful, False otherwise.
        """
        raise NotImplementedError

    def flush(self) -> None:
        """
        Flushes the CPU provider, releasing all allocated CPUs.
        """
        raise NotImplementedError

    def cpu_usage(self) -> dict[str, float]:
        """
        Returns:
            Dict[str, float]: the CPU usage per requester.
        """
        raise NotImplementedError


class UnboundedCPU(Entity):
    """A CPU provider that never refuses requests.

    This provider records fractional allocations per requester but never fails
    a request (useful for simulations that don't model CPU starvation).
    """

    def __init__(
        self,
        max_cpus: float,
        cpu_speed: float,
        observers: list[Observer] | None = None,
    ) -> None:
        """Initialize.

        Args:
            max_cpus (float): Maximum CPUs available (informational).
            cpu_speed (float, optional): CPU speed (Hz) — Defaults to 0.0.
            observers: Optional list of observers to notify on updates.
        """
        self._used_cpus: dict[str, float] = {}
        self._max_cpus = max_cpus
        self._cpu_speed = cpu_speed
        # make a defensive copy so caller can't mutate our list later
        self._observers: list[Observer] = list(observers) if observers else []

    def max_cpus(self) -> float:
        """
        Returns:
            float: Return the configured maximum number of CPUs.
        """
        return self._max_cpus

    def cpu_speed(self):
        """
        Returns:
            float: the configured CPU speed.
        """
        return self._cpu_speed

    def free_cpus(self) -> float:
        """
        Returns:
            float: Return remaining CPUs (may be negative if over-allocated).
        """
        return self._max_cpus - self.used_cpus()

    def used_cpus(self) -> float:
        """
        Returns:
            float: sum of all current allocations (fractional).
        """
        return sum(self._used_cpus.values())

    def request_cpu(self, requester: str, cpus_frac: float, cpu_cycles: int) -> EventExpression:
        """
        Record a CPU allocation for ``requester`` (always succeeds).

        This method requests a CPU allocation of the given size, in CPU fractions.
        The request is never refused, and the requester is always added to the used
        CPUs.

        Args:
            requester (str): The name of the requester of the CPU allocation.
            cpus_frac (float): The size of the CPU allocation to request, in CPU
                fractions.
            cpu_cycles: the number of cpu cycles needed to complete the task

        Returns:
            bool: Always True.
        """
        self._used_cpus[requester] = float(cpus_frac)
        if self._observers is not None:
            for observer in self._observers:
                observer.update(self)

        cycle_duration = cpu_cycles / self._cpu_speed
        return self._schedule_after(cycle_duration, CPU_CYCLES_COMPLETED)

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
        self._used_cpus[requester] = 0
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


class BoundedCPU(Entity):
    """A CPU that, despite having a limited capacity,
       will not fail ever on CPU requests.

    Created to models where we do not want to deal with lack of CPUs.
    """

    class CpuAllocation:
        def __init__(
            self,
            requester: str,
            cpus_requested: float,
            cpu_cycles: int,
            end_type: EventType,
        ):
            self.requester = requester
            self.cpus_requested = cpus_requested
            self.cpus_allocated = cpus_requested
            self.cpu_cycles = cpu_cycles
            self.cycles_remaining = cpu_cycles
            self.end_type = end_type

    def __init__(
        self,
        max_cpus: float,
        cpu_speed: float,
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
        self._used_cpus: dict[str, BoundedCPU.CpuAllocation] = {}
        self._max_cpus = max_cpus
        self._cpu_speed = cpu_speed
        self._observers = observers
        self._last_update_time = DynAASim().current_time
        self._wait(
            EventHandler(self._update_allocations),
            entity=self,
            event_type=CPU_USAGE_CHANGE,
        )

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
        allocated = map(lambda x: x.cpus_allocated, self._used_cpus.values())
        return sum(allocated)

    def _update_cycle_allocations(self, *args, **kwargs):
        """
        Updates the allocations of the bounded CPU,
        with regards to the number of cycles

        It assumes the resource is divided evenly, with a weight that is equal to the
        requested cpu.
        """
        delta_time = DynAASim().current_time - self._last_update_time
        self._last_update_time = DynAASim().current_time

        for allocation in list(self._used_cpus.values()):
            allocation.cycles_remaining -= (
                (allocation.cpus_allocated / allocation.cpus_requested) * delta_time * self._cpu_speed
            )
            if allocation.cycles_remaining <= 0:
                self._schedule_now(allocation.end_type)
                self._used_cpus.pop(allocation.requester)

    def _update_ncpus_allocations(self, *args, **kwargs):
        """
        Updates the allocations of the bounded CPU,
        with regards to the number of cpus

        It assumes the resource is divided evenly, with a weight that is equal to the
        requested cpu.
        """
        total_requested = sum(map(lambda x: x.cpus_requested, self._used_cpus.values()))
        if total_requested == 0:
            return

        fraction = min(1, self._max_cpus / total_requested)
        min_remaining_time = None

        for allocation in self._used_cpus.values():
            allocation.cpus_allocated = allocation.cpus_requested * fraction
            remaining_time = allocation.cycles_remaining / (fraction * self._cpu_speed)
            if min_remaining_time is None or remaining_time < min_remaining_time:
                min_remaining_time = remaining_time

        return self._schedule_after(min_remaining_time, CPU_USAGE_CHANGE)

    def _update_allocations(self, *args, **kwargs):
        """
        Updates the allocations of the bounded CPU.

        It assumes the resource is divided evenly, with a weight that is equal to the
        requested cpu.
        """
        self._update_cycle_allocations()
        return self._update_ncpus_allocations()

    def request_cpu(self, requester: str, cpus_frac: float, cpu_cycles: int) -> bool:
        """
        Requests CPUs from the provider.

        This method requests a CPU allocation of the given size, in CPU fractions.
        The request is never refused, and the requester is always added to the used
        CPUs.

        Args:
            requester (str): The name of the requester of the CPU allocation.
            cpus_frac (float): The size of the CPU allocation to request, in CPU
                fractions.
            cpu_cycles: Total number of cpu cycles needed.

        Returns:
            bool: Always True.
        """
        end_type = EventType("SEGMENT_FINISHED", f"Segment of requester {requester} finished")
        self._update_cycle_allocations()
        self._used_cpus[requester] = BoundedCPU.CpuAllocation(requester, cpus_frac, cpu_cycles, end_type)
        self._update_ncpus_allocations()

        if self._observers is not None:
            for observer in self._observers:
                observer.update(self)

        return EventExpression(source=self, event_type=end_type)

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
        if requester in self._used_cpus:
            self._used_cpus.pop(requester)
            self._update_allocations()
            return True
        return False

    def flush(self) -> None:
        """
        Flushes the CPU provider, releasing all allocated CPUs.
        """
        self._used_cpus = {}

    def cpu_usage(self) -> dict[str, float]:
        """
        Returns the CPU usage per requester.

        Returns:
            dict[str, float]: A dictionary where the keys are the requesters and the
            values are the CPU usage for each requester.
        """
        __cpu_usage = {requester: allocation.cpus_allocated for requester, allocation in self._used_cpus.items()}
        return __cpu_usage
