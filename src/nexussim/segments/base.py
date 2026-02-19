import random
from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import Enum
from typing import Protocol

from pydynaa import Entity, EventExpression, EventHandler, EventType

from nexussim.exceptions import NexussimException


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

    Segments represent blocks of code __activity__ that are bundled together for
    modelling purposes.  It is useful to think about segments as parts of a program for
    which the user can assign a certain behavior and resource usage.

    So for example, a program can have a segment where it basically waits for input
    (e.g. a web request) -- this segment can be characterized by very low cpu and memory
    load, as basically the program is idle or on hold.  After a message arrives, the
    program switches to a different segment to handle the request -- this new segment
    has a higher cpu and memory usage, as it is handling the request.  After the request
    is handled, the program returns to the idle segment to wait for the next request.

    Nexussim provides basic segment model archetypes: example segments for constant load
    or Gaussian load usage.  Users can derive from the Segment base class to model their
    own segment models.

    All segment models have a common basic behavior:

    - A segment can be in one of three states: IDLE, RUNNING, COMPLETED. And the state
      can be obtained via the `state` property.

    - A segment can be executed by calling the `execute` method.  When a segment is
      executed, it transitions to the RUNNING state.  Implementers of segment models can
      define a body for the segment -- the body is a generator of events.  During the
      segment execution, it waits for events produced (in sequence) by the body. When an
      event occurs, the segment resumes, adjusts resource loads, and generates a new
      event to wait for. This process repeats until the segment completes. Upon
      completion, the body generator must return None.  This produces a
      SEGMENT_COMPLETED event (fixed) and throws the segment into the COMPLETED state.

        - If the `execute` method is called while the segment is already in the RUNNING
          state, an exception is raised.

    - A segment can also be stopped by calling the `stop` method.  When a segment is
      stopped, it transitions to the COMPLETED state.  If the segment is already in the
      COMPLETED state, an exception is raised.

    - A segment can be reset by calling the `reset` method.  When a segment is reset, it
      transitions to the IDLE state.
    """

    SEGMENT_COMPLETED = EventType("SEGMENT_COMPLETED", "Event emmited at the end of a segment")

    RESUME_SEGMENT = EventType("RESUME_SEGMENT", "Event that triggers the resume of a segment")

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
        self._id_suffix = hex(random.getrandbits(20))[2:]
        self._handlers = []

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
        return self._parent + "::" + self._id_suffix

    def suffix(self, suffix: str):
        self._id_suffix = suffix

    @property
    def parent(self) -> str:
        return self._parent

    @parent.setter
    def parent(self, parent: str):
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
                self._handlers.append(self._wait_once(resume_handler, expression=wait_event))
            except StopIteration:
                self.stop()
            return

        raise NexussimException("Segment is already completed. Cannot execute.")

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
        for h in self._handlers:
            self._dismiss(h)

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


class EventProvider(Protocol):
    def scheduled_event(self) -> EventExpression:
        """
        Returns an event expression that will be triggered in the future.

        EventProviders are entities that throw events to be observed by a segment.

        Returns:
            EventExpression: An event expression that represents a condition to wait for
            during the segment's execution.
        """
        raise NotImplementedError
