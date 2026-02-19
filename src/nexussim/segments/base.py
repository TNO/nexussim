from abc import ABC, abstractmethod
from enum import Enum

from pydynaa import Entity, EventExpression, EventHandler, EventType
from ulid import ulid
from typing import Generator


class SegmentState(Enum):
    """
    Enumerates the possible states of a segment.
    """

    IDLE = 1
    RUNNING = 2
    COMPLETED = 3


class Segment(ABC, Entity):
    """
    Defines a standard base class to all segments.
    """

    SEGMENT_COMPLETED = EventType(
        "SEGMENT_COMPLETED", "Event emmited at the end of a segment"
    )

    RESUME_SEGMENT = EventType(
        "RESUME_SEGMENT", "Event that triggers the resume of a segment"
    )

    def __init__(self) -> None:
        """
        Initializes the segment.

        State is set to IDLE.

        Returns:
            None
        Raises:
            ValueError: If the body is None
        """
        self._state = SegmentState.IDLE
        self._parent = "<root>"
        self._id = self._parent + "::" + str(ulid())

    @property
    def state(self) -> SegmentState:
        """
        Returns the current state of the segment.

        Returns:
            SegmentState: The current state of the segment.
        """
        return self._state

    @property
    def id(self) -> str:
        return self._id

    @property
    def parent(self) -> str:
        return self._parent

    @parent.setter
    def parent(self, parent: str):
        self._id = self._id.replace(self._parent, parent)
        self._parent = parent

    def execute(self, context: dict) -> None:
        """
        Executes the segment based on its current state.

        If the segment is in the IDLE state, the segment body is initialized and its
        execution is started. The segment transitions to the RUNNING state.

        If the segment is already RUNNING, this function resumes the execution of the
        body of the segment. While new events are yielded, this mechanism continues.

        Upon completion of body execution (no wait event), the segment transitions to
        the COMPLETED state and stops the segment. If the segment is already COMPLETED,
        an exception is raised.

        Args:
            context (dict): A dictionary providing context for the execution.

        Returns:
            None or event: Returns an event to wait on if execution is ongoing,
            otherwise returns None if execution is completed.

        Raises:
            Exception: If the segment is already completed and cannot execute again.
        """

        if self._state == SegmentState.IDLE:
            _seg = self._segment_body(context)
            if not isinstance(_seg, Generator):
                self.stop()
                return
            else:
                self._seg = _seg
                self._state = SegmentState.RUNNING

        if self._state == SegmentState.RUNNING:
            try:
                wait_event = next(self._seg)
                resume_handler = EventHandler(lambda report: self.execute(context))
                self._wait_once(resume_handler, expression=wait_event)
            except StopIteration:
                self.stop()
            return

        raise Exception("Segment is already completed. Cannot execute.")

    def stop(self):
        """
        Stops the segment execution and transition to COMPLETED state.

        This method is used when receiving the StopIteration exception when the
        generator returned by the _segment_body has finished.

        It can be called earlier as a way to stop the execution of the segment.

        It also cares that a SEGMENT_COMPLETED event is thrown.
        """
        self._state = SegmentState.COMPLETED
        self._schedule_now(self.SEGMENT_COMPLETED)
        if hasattr(self, "_seg"):
            del self._seg

    def reset(self):
        """
        Resets the segment to its initial state.

        This method sets the segment's state to IDLE and removes any existing segment
        body generator, effectively resetting the segment. It can be used to
        reinitialize the segment for a new execution.

        Note that during a reset, __no__ SEGMENT_COMPLETED event is thrown.

        It is safe to call reset when the segment is alredy IDLE or COMPLETED.  If the
        segment is reset while in the RUNNING state, there may be side effects (e.g.
        event handlers not being removed, memory leak, or other unexpected behavior).
        """
        self._state = SegmentState.IDLE
        if hasattr(self, "_seg"):
            del self._seg

    @abstractmethod
    def _segment_body(self, context) -> Generator[EventExpression, None, None]:
        """
        Defines the behavior of the segment (as a generator).

        This method should be implemented to yield event expressions based on the
        provided context. It acts as a generator that drives the execution flow of the
        segment.

        At each time, yield one event expression to wait on.  When the event meeting the
        expression is thrown, the segment will resume execution. Once the segment
        completes its execution, it should return None.

        Args:
            context (dict): A context object providing access to the segment's context.

        Yields:
            EventExpression: An event expression that represents a condition to wait for
            during the segment's execution.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        raise NotImplementedError()
