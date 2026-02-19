from typing import Callable

from pydynaa import Entity, EventHandler

from nexussim.context import Context
from nexussim.segments.base import EventProvider, Segment
from nexussim.segments.behaviour import (ConstantLoadSegment,
                                         ConstantLoadWaitSegment,
                                         SampledLoadSegment)
from nexussim.segments.control import (ChoiceSegment, ConcurrentSegment,
                                       RepeatSegment, SequenceSegment,
                                       WhileSegment, forever)


class Program(Entity):
    """A Program represents a software component that can be modeled by its diverse
    segments."""

    def __init__(
        self,
        name: str,
        body: Segment,
        cpu_load: float = 0.0,
        mem_load: float = 0.0,
        default_on_finish: Callable[[Context], None] = None,
    ):
        self.name = name
        self.body = body
        self.body.parent = self.name
        self.base_cpu_load = cpu_load
        self.base_mem_load = mem_load
        self._on_finish_handler = None

    def execute(self, context):
        """
        Execute the program's body segment in the given context.

        The program name is set as the parent of the body segment.

        The body segment is executed in the given context and the program
        waits for the body segment's completion.

        When the body segment completes, the program calls its
        `default_on_finish_callback` with the handle report and the context.

        Args:
            context: The context in which to execute the program's body.
        """
        self.body.reset()
        context.compute.cpu.request_cpu(self.name, self.base_cpu_load)
        self.body.execute(context)
        self._on_finish_handler = self._wait_once(
            EventHandler(
                lambda h_report: self.default_on_finish_callback(h_report, context)
            ),
            entity=self.body,
            event_type=self.body.SEGMENT_COMPLETED,
        )

    def stop(self):
        self.body.stop()
        if self._on_finish_handler:
            self._dismiss(self._on_finish_handler)

    def default_on_finish_callback(self, h_report, context):
        # TODO: This is just a hacked example. Think about something more instructive.
        context.compute.cpu.release_cpu(self.name)


def constant_load_segment(
    cpu_load: float = 1.0, mem_load: float = 0.0, duration: float = 1.0, cycles: int = 0
):
    """
    Factory for creating a constant CPU load over a specified duration.

    Args:
        cpu_load (float): The amount of CPU load requested, as a fraction of total
        capacity.

        mem_load (float): The amount of memory load requested, in bytes.

        duration (float): The duration of the segment in seconds.

        cycles (int): The number of cycles to run.

    Returns:
        Segment: A Segment that represents a constant CPU load over the specified
        duration.
    """
    return ConstantLoadSegment(
        cpu_load=cpu_load, mem_load=mem_load, duration=duration, cycles=cycles
    )


def constant_load_wait_segment(
    wait_on: EventProvider, cpu_load: float = 1.0, mem_load: float = 0.0
) -> ConstantLoadWaitSegment:
    """
    Factory for creating a segment
    that waits on an event provider and requests a constant CPU and memory load.

    Args:
        wait_on (EventProvider): The event provider to wait on.

        cpu_load (float, optional):
        The CPU load requested, as a fraction of total capacity. Defaults to 1.0.

        mem_load (float, optional):
        The amount of memory load requested, in bytes. Defaults to 0.0.

    Returns:
        ConstantLoadWaitSegment:
        A Segment that represents waiting
        on an event and requesting a constant CPU and memory load.
    """
    return ConstantLoadWaitSegment(
        wait_on=wait_on, cpu_load=cpu_load, mem_load=mem_load
    )


def sampled_load_segment(cpu_load_sampler, freq, duration):
    """
    Factory for creating a sampled CPU load over a specified duration.

    Args:
        cpu_load_sampler (Generator): A generator function that samples CPU load values.
        freq (float): The frequency at which the CPU load is sampled, in Hz.
        duration (float): The duration of the segment in seconds.

    Returns:
        Segment: A Segment that represents a sampled CPU load over the specified
        duration.
    """
    return SampledLoadSegment(
        cpu_load_sampler=cpu_load_sampler, freq=freq, duration=duration
    )


def sequence(*segments):
    """
    Factory for creating a sequence of segments.

    Args:
        *segments (Segment): Segments to be executed in sequence.

    Returns:
        Segment: A Segment that represents the sequence of the given segments.
    """
    return SequenceSegment(segments=segments)


def repeat(segment, times):
    """
    Factory for creating a RepeatSegment which repeats a given segment a specified
    number of times.

    Args:
        segment (Segment): The segment to be repeated. times (int): The number of
        repetitions for the segment.

    Returns:
        RepeatSegment: A Segment that represents the repeated execution of the given
        segment.
    """
    return RepeatSegment(segment=segment, times=times)


def concurrent(*segments):
    """
    Factory for creating a ConcurrentSegment which executes multiple segments
    concurrently.

    Args:
        *segments (Segment): Segments to be executed concurrently.

    Returns:
        Segment: A Segment that represents the concurrent execution of the given
        segments.
    """
    return ConcurrentSegment(segments=segments)


def choice(*segments, weights=None):
    """
    Factory for creating a ChoiceSegment which chooses a segment at random based on a
    given set of weights.

    Args:
        *segments (Segment):
        Segments to be considered for choosing a segment.

        weights (list[float]):
        Weights associated with the choice of each segment.
    """
    return ChoiceSegment(segments=segments, weights=weights)


def while_do(segment, condition: Callable[[Context], bool] = forever):
    return WhileSegment(segment=segment, condition=condition)
