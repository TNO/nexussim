import sys
import warnings
from collections.abc import Generator

from pydynaa import EventHandler

from nexussim.compute import Compute
from nexussim.samplers import gaussian_load_sampler
from nexussim.segments.base import EventProvider, Segment

warnings.simplefilter("always", DeprecationWarning)  # force deprecation warnings


class SampledLoadSegment(Segment):
    def __init__(
        self,
        cpu_load_sampler: Generator = None,
        freq: float = 1.0,
        duration: float = 10.0,
    ):
        """
        Initializes the SampledLoadSegment with a CPU load sampler, frequency, and
        duration.

        Args:
            cpu_load_sampler (Generator, optional):
                A generator function that samples CPU load values. Sampler __must__ take
                care of maximal load and minimal load constraints.Defaults to a Gaussian
                load sampler.

            freq (float, optional):
                The frequency at which the CPU load is sampled, in Hz. Must be greater
                than 0. Defaults to 1.0.

            duration (float, optional):
                The duration of the segment in seconds. Must be greater than 0. Defaults
                to 1.0.

        Raises:
            ValueError:
                If the freq parameter is less than or equal to 0. ValueError: If the
            duration parameter is less than or equal to 0.
        """
        if cpu_load_sampler is None:
            cpu_load_sampler = gaussian_load_sampler()
        if freq < 0.0:
            raise ValueError(f"freq must not be negative, but got {freq}.")
        if duration < 0.0:
            raise ValueError(f"duration must not be negative, but got {duration}.")

        super().__init__()
        self._cpu_load_sampler = cpu_load_sampler
        if freq <= 0:
            raise ValueError("freq parameter (frequency, in Hz) must be greater than 0.")
        self._freq = freq
        if duration <= 0:
            raise ValueError("duration parameter must be greater than 0.")
        self._duration = duration
        warnings.warn(f"Should not use the {type(self)}", DeprecationWarning, stacklevel=2)

    def _segment_body(self, context):
        """
        This method implements the behavior of SampledLoadSegment.

        It samples from the cpu_load_sampler and requests the sampled CPU load
        from the context's CPU for the specified duration, at the specified
        frequency. The sampler is called every 1/freq seconds. The segment
        transitions to the RUNNING state and keeps requesting CPU resources until
        the duration is over.

        At the end of the duration, the segment transitions to the COMPLETED
        state and releases the previously requested CPU resources.

        Args:
            context (dict): A context object providing access to the segment's
                context.

        Yields:
            EventExpression: An event expression that represents a condition to
                wait for during the segment's execution.
        """
        sample_time = 0.0
        dt = 1.0 / self._freq
        while sample_time < self._duration:
            cpu_load = self._cpu_load_sampler()
            context.compute.cpu.request_cpu(self.id, cpu_load, self._freq * self._duration)
            yield self._schedule_after(dt, self.RESUME_SEGMENT)
            sample_time += dt
        # Cleaning resources
        context.compute.cpu.release_cpu(self.id)


class ConstantLoadSegment(Segment):
    def __init__(
        self,
        cpu_load: float = 1.0,
        mem_load: float = 0.0,
        duration: float = 0.0,
        cycles: int = 0,
        message: dict = None,
    ):
        """
        Initializes the ConstantLoadSegment with a CPU load and duration.
        Note: we expect either a durationor a number of cycles but not both

        Args:
            cpu_load (float, optional): The CPU load as a fraction of total CPU
            capacity.
                Defaults to 1.0.
            mem_load (float, optional): The memory load in bytes.
            cycles (int, optional): The number of cycles to run. Defaults to 0.
            message (dict) is an optional dict describing a message to emit
            when segment completes. Expected keys (as produced by the scene parser):
            - "msg_size": size in bytes
            - "msg_destination": a string or list of destinations
        Remarks:
            _duration_ and _cycles_ add up.

        """
        if cpu_load < 0.0:
            raise ValueError(f"cpu_load must not be negative, but got {cpu_load}.")
        if mem_load < 0.0:
            raise ValueError(f"mem_load must not be negative, but got {mem_load}.")
        if duration < 0.0:
            raise ValueError(f"duration must not be negative, but got {duration}.")
        if cycles < 0.0:
            raise ValueError(f"cycles must not be negative, but got {cycles}.")
        if (duration == 0.0) == (cycles == 0):
            raise ValueError("Must have either a durationor a number of cycles (not both)")
        self._cpu_load = cpu_load
        self._mem_load = mem_load
        self._duration = duration
        self._cycles = cycles
        self._message = message or {}

        super().__init__()

    @property
    def cpu_load(self):
        """
        Returns the CPU load for the segment.

        This property provides the CPU load that is requested by this segment during
        its execution.

        Returns:
            float: The CPU load as a fraction of total CPU capacity.
        """

        return self._cpu_load

    @property
    def mem_load(self):
        """
        Returns the memory load for the segment.

        This property provides the memory load in bytes that is requested by this
        segment during its execution.

        Returns:
            float: The memory load in bytes.
        """
        return self._mem_load

    @property
    def duration(self):
        """
        Returns the duration for which the segment runs.

        Returns:
            float: The duration in seconds.
        """

        return self._duration

    @property
    def cycles(self):
        """
        Returns the number of cycles for the segment.

        This property provides the number of cycles this segment is supposed to run.

        Returns:
            int: The number of cycles.
        """
        return self._cycles

    def _segment_body(self, context):
        """
        This method requests constant CPU and Memory resources for the segment, then
        yields an event expression to wait for the specified duration. After the
        duration has elapsed, the segment is resumed.

        Once the segment completes its execution, it releases the CPU resources.

        Args:
            context (dict): A context object providing access to the compute resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the duration of the segment's execution.
        """
        compute: Compute = context.compute
        # if we had only been given a duration, we compute the number of cycles to run
        # based on the CPU speed.
        cycles = self._cycles if self._cycles != 0 else compute.cpu.cpu_speed() * self._duration

        addr = None
        if self._mem_load is not None and self._mem_load > 0:
            addr = compute.memory.malloc(self._mem_load, self.id)

        # Ensure CPU is always released and memory freed when the segment
        # generator finishes or an error occurs. We keep the original order
        # (request -> resume -> release -> emit -> free) but move the cleanup
        # into a finally block so it always runs.
        try:
            yield compute.cpu.request_cpu(self.id, self._cpu_load, cycles)

            yield self._schedule_now(self.RESUME_SEGMENT)
        finally:
            # Cleaning resources
            context.compute.cpu.release_cpu(self.id)

            # Emit message (if configured) after the segment completes. Failures
            # in messaging must not prevent resource cleanup.
            if self._message:
                if not hasattr(context.compute, "context") or not hasattr(context.compute.context, "router"):
                    raise RuntimeError("Compute must have a context with a router to emit messages.")
                router = context.compute.context.router
                router.emit_message(self.id, self.parent, self._message, context.compute)

            # Free any allocated memory for this segment.
            if addr is not None:
                try:
                    compute.memory.free(addr)
                except Exception:
                    # best-effort free; don't raise from here
                    pass


class SleepSegment(Segment):
    def __init__(self, duration: float, mem_load: float = 0.0):
        """Initializes the SleepSegment with a memory load.

        Args:
            duration (float): The duration of the segment in seconds.
            mem_load (float, optional): The memory load in bytes.
        """
        if mem_load < 0.0:
            raise ValueError(f"mem_load must not be negative, but got {mem_load}.")
        if duration < 0.0:
            raise ValueError(f"duration must not be negative, but got {duration}.")
        self._duration = duration
        self._mem_load = mem_load
        super().__init__()

    @property
    def mem_load(self):
        """The memory load in bytes requested by this segment."""
        return self._mem_load

    @property
    def duration(self):
        """The duration in seconds for which the segment runs."""
        return self._duration

    def _segment_body(self, context):
        """Wait for a fixed duration, freeing any allocated memory when the wait ends.

        Allocates memory (if requested), schedules a resume event after
        `self.duration`, and attaches a one-shot handler to free the memory
        when that resume event fires.
        """
        compute: Compute = context.compute
        addr = None
        if self._mem_load is not None and self._mem_load > 0:
            addr = compute.memory.malloc(self._mem_load, self.id)

        # schedule the resume event and also attach a one-off handler to free memory
        wait_ev = self._schedule_after(self.duration, self.RESUME_SEGMENT)
        if addr is not None:
            # capture addr in a default parameter to avoid late-binding
            self._wait_once(
                EventHandler(lambda report, _addr=addr: compute.memory.free(_addr)),
                expression=wait_ev,
            )

        yield wait_ev


# Backwards-compatible wrapper: warn only when the old name is instantiated.
class WaitSegment(SleepSegment):
    def __init__(self, *args, **kwargs):
        warnings.warn("WaitSegment is deprecated, use SleepSegment instead", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class ConstantLoadWaitSegment(Segment):
    def __init__(
        self,
        wait_on: EventProvider = None,
        cpu_load: float = 1.0,
        mem_load: float = 0.0,
    ):
        """
        Initializes the ConstantLoadWaitSegment with a CPU and memory load and event to
        wait upon.

        Args:
            wait_on (EventProvider):
                The event provider to wait upon.
            cpu_load (float, optional):
                The CPU load as a fraction of total CPU capacity.Defaults to 1.0.
            mem_load (float, optional):
                The memory load in bytes.
        Remarks:
            _duration_ and _cycles_ add up.

        """
        if cpu_load < 0.0:
            raise ValueError(f"cpu_load must not be negative, but got {cpu_load}.")
        if mem_load < 0.0:
            raise ValueError(f"mem_load must not be negative, but got {mem_load}.")
        self._event_provider = wait_on
        self._cpu_load = cpu_load
        self._mem_load = mem_load
        super().__init__()
        warnings.warn(f"Should not use the {type(self)}", DeprecationWarning, stacklevel=2)

    @property
    def cpu_load(self):
        """
        Returns the CPU load for the segment.

        This property provides the CPU load that is requested by this segment during
        its execution.

        Returns:
            float: The CPU load as a fraction of total CPU capacity.
        """

        return self._cpu_load

    @property
    def mem_load(self):
        """
        Returns the memory load for the segment.

        This property provides the memory load in bytes that is requested by this
        segment during its execution.

        Returns:
            float: The memory load in bytes.
        """
        return self._mem_load

    def _segment_body(self, context):
        """
        This method requests constant CPU and Memory resources for the segment, then
        collects an event to be observed (from an event provider). The segment is
        resumed after the event is produced by the provider.

        Only waits for the first event to be thrown by the event provider.

        Once the segment completes its execution, it releases the CPU resources.

        Args:
            context (dict): A context object providing access to the compute resources.

        Yields:
            EventExpression: An event expression that represents the wait condition that
            resumes the segment's execution.
        """
        compute: Compute = context.compute
        addr = None
        if self._mem_load is not None and self._mem_load > 0:
            addr = compute.memory.malloc(self._mem_load, self.id)

        compute.cpu.request_cpu(self.id, self.cpu_load, cpu_cycles=sys.maxsize)
        yield self._event_provider.scheduled_event()

        # Cleaning resources
        context.compute.cpu.release_cpu(self.id)
        if addr is not None:
            compute.memory.free(addr)
