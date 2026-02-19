import random
from collections.abc import Callable, Generator, Iterable

import pydynaa as pd

from nexussim.context import Context
from nexussim.segments.base import Segment


class SequenceSegment(Segment):
    """A segment that executes a list of segments in sequence."""

    def __init__(self, segments: Iterable[Segment]):
        super().__init__()
        self._segments = segments
        for segment in self._segments:
            segment.parent = self.id

    @property
    def segments(self):
        """Returns the list of segments in the sequence segment."""
        return self._segments

    @Segment.parent.setter
    def parent(self, parent: str):
        """Sets the parent of the concurrent segment.
        Updates child segments.
        """
        self._parent = parent
        for segment in self._segments:
            segment.parent = self.id

    def _segment_body(self, context: Context) -> Generator[pd.EventExpression, None, None]:
        """
        This method executes the segments in ordered sequence.

        It resets each segment to its initial state, executes it and waits for its
        completion before moving on to the next segment.

        Args:
            context (Context): A context object providing access to the compute
            resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the completion of each segment's execution.
        """
        for segment in self._segments:
            segment.reset()  # Always start with a fresh segment.
            # TODO: Add into to context here?
            segment.execute(context)
            wait_on = pd.EventExpression(segment, segment.SEGMENT_COMPLETED)
            yield wait_on


class WhileSegment(Segment):
    """A segment that executes a given segment while a given condition is True."""

    def __init__(self, segment: Segment, condition: Callable[[Context], bool]) -> None:
        """
        Initializes a WhileSegment with a segment.

        A while segment mimics a while loop. It executes a segment while a condition is
        True. The condition is inspected immediatelly before each execution of the
        segment. The segment is executed until completion and then the condition is
        re-evaluated.

        condition is a function that takes a context (Context) as input and returns a
        boolean value.

        Args:
            segment (Segment): The segment to be executed in the while segment.
        """
        super().__init__()
        self._segment = segment
        self._segment.parent = self.id
        self._condition = condition

    def _segment_body(self, context: Context) -> Generator[pd.EventExpression, None, None]:
        while self._condition(context):
            self._segment.reset()
            self._segment.execute(context)
            wait_on = pd.EventExpression(self._segment, self._segment.SEGMENT_COMPLETED)
            yield wait_on

    @property
    def segment(self):
        """Returns the segment to be executed in the while segment."""
        return self._segment

    @Segment.parent.setter
    def parent(self, parent: str):
        self._parent = parent
        self._segment.parent = self.id


class RepeatSegment(Segment):
    """A segment that repeats a given segment a number of times."""

    def __init__(self, segment: Segment, times: int) -> None:
        """
        Initializes a RepeatSegment with a segment and the number of times to repeat.

        Args:
            segment (Segment): The segment to be repeated.
            times (int): The number of times the segment is to be repeated.

        Raises:
            ValueError: If the number of times is negative.
        """
        super().__init__()
        self._segment = segment
        self._segment.parent = self.id
        if times < 0:
            raise ValueError("Number of times must be positive")
        self._times = times

    @property
    def segment(self):
        """Returns the segment to be repeated."""
        return self._segment

    # Overrides the parent property setter
    @Segment.parent.setter
    def parent(self, parent: str):
        self._parent = parent
        self._segment.parent = self.id

    def _segment_body(self, context: Context) -> Generator[pd.EventExpression, None, None]:
        """
        This method executes the given segment a number of times.

        It resets the segment after each execution and waits for its completion before
        moving on to the next iteration.

        Args:
            context (Context): A context object providing access to the compute
            resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the completion of each segment's execution.
        """
        for _ in range(self._times):
            self._segment.reset()
            self._segment.execute(context)
            wait_on = pd.EventExpression(self._segment, self._segment.SEGMENT_COMPLETED)
            yield wait_on


class ConcurrentSegment(Segment):
    """A segment that executes a list of segments concurrently."""

    def __init__(self, segments: list[Segment]):
        super().__init__()
        self._segments = segments
        if not segments:
            raise ValueError("ConcurrentSegment must have at least one segment")
        for segment in self._segments:
            segment.parent = self.id

    @Segment.parent.setter
    def parent(self, parent: str):
        """Sets the parent of the concurrent segment.
        Updates child segments.
        """
        self._parent = parent
        for segment in self._segments:
            segment.parent = self.id

    def _segment_body(self, context: Context) -> Generator[pd.EventExpression, None, None]:
        """
        This method executes all segments in the list concurrently.

        It resets each segment in the list and executes them all in parallel. The method
        waits for all segments to complete before returning.

        Args:
            context (Context): A context object providing access to the compute
            resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the completion of all segments' execution.
        """
        wait_events = None
        for segment in self._segments:
            segment.reset()
            segment.execute(context)
            wait_events = pd.EventExpression(segment, segment.SEGMENT_COMPLETED) & wait_events

        yield wait_events

    @property
    def segments(self):
        """Returns the list of segments in the concurrent segment."""
        return self._segments


class ChoiceSegment(Segment):
    """A segment that chooses one of a list of segments to execute."""

    def __init__(self, segments: list[Segment], weights: list[float] = None):
        """
        Initializes a ChoiceSegment with a list of segments.

        Args:
            segments (list[Segment]): A list of segments to choose from.
            weights (list[float], optional): A list of weights associated with each
            segment. Defaults to None.

        Raises:
            ValueError: If the list of segments is empty.
        """
        super().__init__()
        self._segments = segments
        if not segments:
            raise ValueError("ChoiceSegment must have at least one segment")
        for segment in self._segments:
            segment.parent = self.id
        self._weights = weights

    @Segment.parent.setter
    def parent(self, parent: str):
        """Sets the parent of the concurrent segment.
        Updates child segments.
        """
        self._parent = parent
        for segment in self._segments:
            segment.parent = self.id

    def _segment_body(self, context: Context) -> Generator[pd.EventExpression, None, None]:
        """
        This method executes one segment in the list randomly selected.

        It resets the randomly selected segment and executes it. The method waits for
        the segment to complete before returning.

        Args:
            context (Context): A context object providing access to the compute
            resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the completion of the randomly chosen segment's execution.
        """
        segment_choice = random.choices(population=self._segments, weights=self._weights, k=1)[0]
        segment_choice.reset()
        segment_choice.execute(context)
        wait_on = pd.EventExpression(segment_choice, segment_choice.SEGMENT_COMPLETED)
        yield wait_on

    @property
    def segments(self):
        """Returns the list of segments in the choice segment."""
        return self._segments


# Utility condition functions
def forever(_: Context) -> bool:
    """A condition function that always returns True."""
    return True
