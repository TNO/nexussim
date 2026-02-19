from nexussim.context import Context
from nexussim.segments.base import Segment


import pydynaa as pd


from typing import Callable, Generator


class SequenceSegment(Segment):
    def __init__(self, segments: list[Segment]):
        super().__init__()
        self._segments = segments
        for segment in self._segments:
            segment._parent = self._id

    @property
    def segments(self):
        return self._segments

    def _segment_body(
        self, context: Context
    ) -> Generator[pd.EventExpression, None, None]:
        """
        This method executes the segments in ordered sequence.

        It resets each segment to its initial state, executes it and waits for its
        completion before moving on to the next segment.

        Args:
            context (Context): A context object providing access to the compute resources.

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
    def __init__(self, segment: Segment, condition: Callable[[Context], bool]) -> None:
        """
        Initializes a WhileSegment with a segment.

        A while segment mimics a while loop. It executes a segment while a condition is
        True. The condition is inspected immediatelly before each execution of the
        segment. The segment is executed until completion and then the condition is
        re-evaluated.

        condition is a function that takes a context (Context) as input and returns
        a boolean value.

        Args:
            segment (Segment): The segment to be executed in the while segment.
        """
        super().__init__()
        self._segment = segment
        self._segment._parent = self._id
        self._condition = condition

    def _segment_body(
        self, context: Context
    ) -> Generator[pd.EventExpression, None, None]:
        while self._condition(context):
            self._segment.reset()
            self._segment.execute(context)
            wait_on = pd.EventExpression(self._segment, self._segment.SEGMENT_COMPLETED)
            yield wait_on


class RepeatSegment(Segment):
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
        self._segment._parent = self._id
        if times < 0:
            raise ValueError("Number of times must be positive")
        self._times = times

    def _segment_body(
        self, context: Context
    ) -> Generator[pd.EventExpression, None, None]:
        """
        This method executes the given segment a number of times.

        It resets the segment after each execution and waits for its completion before
        moving on to the next iteration.

        Args:
            context (Context): A context object providing access to the compute resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the completion of each segment's execution.
        """
        for _ in range(self._times):
            self._segment.reset()
            self._segment.execute(context)
            wait_on = pd.EventExpression(self._segment, self._segment.SEGMENT_COMPLETED)
            yield wait_on


## Utility condition functions
def forever(context: Context) -> bool:
    return True
