from typing import Protocol

from pydynaa import DynAASim, Entity, EventHandler, EventType

from nexussim.exceptions import NexussimException

NETWORK_TRANSMISSION_COMPLETED = EventType("NETWORK_TRANSMISSION_COMPLETED", "Network transmission completed")
NETWORK_TRANSMISSION_STARTED = EventType("NETWORK_TRANSMISSION_STARTED", "Network transmission started")


class NetworkProvider(Protocol):
    def max_bandwidth(self) -> float:
        """
        Returns the maximum bandwidth.

        This method should be implemented to return the maximum bandwidth available
        for allocation.

        Returns:
            float: The maximum bandwidth.
        """
        raise NotImplementedError

    def bandwidth(self) -> float:
        """
        Returns the current bandwidth.

        This method should be implemented to return the current bandwidth available
        for allocation.

        Returns:
            float: The current bandwidth.
        """
        raise NotImplementedError

    def latency(self) -> float:
        """
        Returns the current latency.

        This method should be implemented to return the current latency available
        for allocation.

        Returns:
            float: The current latency.
        """
        raise NotImplementedError

    def connect(self, component: str) -> bool:
        """
        Connects a component to the network.

        This method should be implemented to connect a component to the network.

        Args:
            component (str): The component to connect.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        raise NotImplementedError

    def disconnect(self, component: str) -> bool:
        """
        Disconnects a component from the network.

        This method should be implemented to disconnect a component from the network.

        Args:
            component (str): The component to disconnect.

        Returns:
            bool: True if the disconnection was successful, False otherwise.
        """
        raise NotImplementedError

    def send(self, from_: str, to_: str, data_descriptor: dict):
        """Models the sending of data over the network

        Args:
            from_ (str): The component sending the data
            to_ (str): The component receiving the data
            data_descriptor (dict): A descriptor of the data characteristics to be sent
        """
        raise NotImplementedError

    def peers(self):
        """
        Returns a list of component names currently connected to the network.

        Returns:
            list: A copy of the list of connected components.
        """
        raise NotImplementedError


class BaseNetwork(Entity, NetworkProvider):
    def __init__(self, max_bandwidth: float = 1.0, max_latency: float = 1.0, congestion_factor: float = 0.1):
        """
        Initializes a BaseNetwork.

        Args:
            max_bandwidth (float, optional, in bytes/second):
                The maximum bandwidth of the network. Defaults to 1.0 byte/second.
            max_latency (float, optional in seconds):
                The maximum latency of the network. Defaults to 1.0.
            congestion_factor (float, optional):
                Scales queuing-delay inflation under load. When > 0, each in-flight
                transmission at send time adds ``congestion_factor * base_latency``
                to the effective latency:
                ``effective_latency = base_latency * (1 + congestion_factor * N)``.
                Defaults to 0.1.
        """
        self._peers = set()
        self._max_bandwidth = max_bandwidth
        self._max_latency = max_latency
        self._congestion_factor = congestion_factor
        self._inflight_transmissions = {}
        self._transmission_sequence = 0

    def connect(self, peer: str) -> bool:
        """
        Connects a component to the network.

        Args:
            peer (str): The component to connect.

        Returns:
            bool: True if the connection was successful, False if a component with the
            same name is already connected.
        """
        if peer in self.peers:
            return False
        self._peers.add(peer)
        return True

    def disconnect(self, peer: str) -> bool:
        """
        Disconnects a component from the network.

        Args:
            peer (str): The component to disconnect.

        Returns:
            bool: True if the disconnection was successful.
        """
        try:
            self._peers.remove(peer)
            return True
        except KeyError:
            return False

    def send(self, from_: str, to_: str, data_descriptor: dict):
        """
        Models the sending of data over the network.

        Each active transmission shares the network's max_bandwidth equally. New
        transmissions join immediately and trigger a bandwidth redistribution among
        all in-flight transfers.

        If ``data_descriptor`` is None or the payload size is zero or less, a
        ``NETWORK_TRANSMISSION_COMPLETED`` event is fired immediately (the message
        is considered trivially sent).

        Raises ``NexussimException`` if ``max_bandwidth`` is zero or less, since the
        network has no capacity to ever transmit anything.

        Args:
            from_ (str): The component sending the data.
            to_ (str): The component receiving the data.
            data_descriptor (dict): A descriptor of the data characteristics to be sent.

        Raises:
            NexussimException: If the network's max_bandwidth is zero or less.
        """
        if self._max_bandwidth <= 0:
            raise NexussimException(
                f"Cannot send: network max_bandwidth is {self._max_bandwidth}, no capacity to transmit."
            )

        size = float((data_descriptor or {}).get("size", 0.0) or 0.0)
        if size <= 0:
            self._schedule_after(0, NETWORK_TRANSMISSION_COMPLETED)
            return

        effective_latency = self._max_latency * (1 + self._congestion_factor * len(self._inflight_transmissions))
        self._wait_once(
            EventHandler(lambda report: self._start_transmission(size)),
            expression=self._schedule_after(effective_latency, NETWORK_TRANSMISSION_STARTED),
            entity=self,
        )

    def _start_transmission(self, size: float):
        self._transmission_sequence += 1
        transmission_id = self._transmission_sequence
        now = self._now()
        self._inflight_transmissions[transmission_id] = {
            "remaining_size": size,
            "ready_at": now + self._max_latency,
            "handler": None,
            "used_bandwidth": 0.0,
            "last_update_at": now,
        }
        self._reschedule_transmissions(now)

    def _now(self) -> float:
        return DynAASim().current_time

    def _complete_transmission(self, transmission_id: int):
        self._inflight_transmissions.pop(transmission_id, None)
        self._reschedule_transmissions(self._now())

    def _reschedule_transmissions(self, now: float):
        shared_bandwidth = self._max_bandwidth / max(1, len(self._inflight_transmissions))
        for transmission_id, transmission in list(self._inflight_transmissions.items()):
            if transmission["handler"] is not None:
                self._dismiss(transmission["handler"])
            tansmitted_since_last_update = (now - transmission["last_update_at"]) * transmission["used_bandwidth"]

            transmission["remaining_size"] = max(0.0, transmission["remaining_size"] - tansmitted_since_last_update)
            if transmission["remaining_size"] <= 0.0:
                self._complete_transmission(transmission_id)
                continue

            transmission["last_update_at"] = now
            # Schedule transmission without adding latency again (latency already paid in send())
            transmission_time = transmission["remaining_size"] / shared_bandwidth
            transmission["used_bandwidth"] = shared_bandwidth
            transmission["handler"] = self._wait_once(
                EventHandler(
                    lambda report, _transmission_id=transmission_id: self._complete_transmission(_transmission_id)
                ),
                expression=self._schedule_after(transmission_time, NETWORK_TRANSMISSION_COMPLETED),
                entity=self,
            )

    @property
    def peers(self) -> list:
        """
        A list of components currently connected to the network.

        Returns:
            list: A copy of the list of connected components.
        """
        return list(self._peers)

    @property
    def max_bandwidth(self) -> float:
        """
        The maximum bandwidth of the network.

        Returns:
            float: The maximum bandwidth in bytes/second.
        """
        return self._max_bandwidth

    @property
    def latency(self) -> float:
        """
        The current latency of the network.

        Returns:
            float: The current latency in seconds.
        """
        return self._max_latency

    @property
    def bandwidth(self) -> float:
        """
        The current bandwidth of the network.

        Returns:
            float: The current bandwidth in bytes/second.
        """
        used_bandwidth = sum(transmission["used_bandwidth"] for transmission in self._inflight_transmissions.values())
        return max(0.0, self._max_bandwidth - used_bandwidth)
