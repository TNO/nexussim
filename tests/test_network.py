import pydynaa as pd
import pytest

from nexussim.exceptions import NexussimException
from nexussim.network import NETWORK_TRANSMISSION_COMPLETED, BaseNetwork


@pytest.fixture
def defaultBaseNetwork():
    return BaseNetwork()


@pytest.fixture
def configuredBaseNetwork():
    return BaseNetwork(max_bandwidth=100, max_latency=101)


def test_connect(defaultBaseNetwork):
    assert defaultBaseNetwork.connect("test")
    assert not defaultBaseNetwork.connect("test")
    assert defaultBaseNetwork.peers == ["test"]


def test_disconnect(defaultBaseNetwork):
    assert defaultBaseNetwork.connect("test")
    assert defaultBaseNetwork.disconnect("test")
    assert defaultBaseNetwork.peers == []


def test_send(defaultBaseNetwork):
    defaultBaseNetwork.send("test", "test", None)


def test_send_zero_size_fires_completed_event():
    pd.DynAASim().reset()
    dynaa_sim = pd.DynAASim()
    net = BaseNetwork(max_bandwidth=100, max_latency=1)
    completed = {"count": 0}

    def on_completed(_report):
        completed["count"] += 1

    pd.Entity()._wait(pd.EventHandler(on_completed), entity=net, event_type=NETWORK_TRANSMISSION_COMPLETED)

    # Zero-size messages are trivially sent and fire a completed event immediately.
    net.send("sender", "receiver", {"size": 0})
    net.send("sender", "receiver", None)

    dynaa_sim.run()

    assert completed["count"] == 2
    assert dynaa_sim.current_time == 0
    assert len(net._inflight_transmissions) == 0
    assert net.bandwidth == 100  # No bandwidth consumed


def test_send_raises_when_max_bandwidth_is_zero():
    net = BaseNetwork(max_bandwidth=0, max_latency=1)
    with pytest.raises(NexussimException):
        net.send("sender", "receiver", {"size": 100})


def test_send_temporarily_reduces_bandwidth():
    pd.DynAASim().reset()
    dynaa_sim = pd.DynAASim()
    net = BaseNetwork(max_bandwidth=100, max_latency=1)

    assert net.bandwidth == 100

    net.send("sender", "receiver", {"size": 50})

    assert net.bandwidth == 100  # Bandwidth should not be reduced immediately after sending. (due to latency)

    dynaa_sim.run(1.0)
    assert net.bandwidth == 100  # Bandwidth should still not be reduced during latency period.
    assert dynaa_sim.current_time == pytest.approx(1.0)  # Verify latency timing

    dynaa_sim.run(1.001)
    assert net.bandwidth == 0  # Bandwidth should be fully reserved for the transmission after latency has passed.

    dynaa_sim.run()
    assert dynaa_sim.current_time == 1.5
    assert net.bandwidth == 100


def test_send_two_transfers_share_bandwidth_then_handover_to_remaining_transfer():
    pd.DynAASim().reset()
    dynaa_sim = pd.DynAASim()
    net = BaseNetwork(max_bandwidth=100, max_latency=0.1)
    assert net.bandwidth == 100
    assert net.latency == 0.1  # Base latency is 0.1 seconds

    # Start a larger transfer, then start a second one while the first is active.
    net.send("sender", "receiver", {"size": 100})
    assert len(net._inflight_transmissions) == 0  # Latency should prevent the transfer from starting immediately.

    dynaa_sim.run(0.101)
    assert len(net._inflight_transmissions) == 1
    assert dynaa_sim.current_time == pytest.approx(0.101)  # Verify latency timing (0.1s latency)
    assert net.bandwidth == 0  # All bandwidth reserved for first transfer

    dynaa_sim.run(0.6)
    net.send("sender", "receiver", {"size": 80})

    # Both transfers should be active and sharing the bandwidth, so neither should complete until 1.4s.
    dynaa_sim.run(1.0)
    assert len(net._inflight_transmissions) == 2
    assert net.bandwidth == 0  # All bandwidth shared between two transfers

    # The larger transfer should complete first at 1.5s (0.1 latency), leaving only the smaller one.
    dynaa_sim.run(1.501)
    assert len(net._inflight_transmissions) == 1
    assert dynaa_sim.current_time == pytest.approx(1.501)  # Verify timing relative to latency
    assert net.bandwidth == 0  # Remaining transfer still using all available bandwidth

    # With default congestion_factor=0.1, the second send has 0.11s latency,
    # so the final completion shifts slightly to 2.31s.
    dynaa_sim.run()
    assert dynaa_sim.current_time == pytest.approx(2.31)
    assert net.latency == 0.1  # Latency returns to base
    assert net.bandwidth == 100


def test_max_bandwidth(configuredBaseNetwork):
    # In the BaseNetwork, bandwidth == max_bandwidth
    assert configuredBaseNetwork.bandwidth == 100
    assert configuredBaseNetwork.max_bandwidth == 100
    assert configuredBaseNetwork.latency == 101


def test_bandwidth(configuredBaseNetwork):
    assert configuredBaseNetwork.bandwidth == 100
    assert configuredBaseNetwork.max_bandwidth == 100
    assert configuredBaseNetwork.latency == 101


def test_connect_duplicate_peer_returns_false():
    net = BaseNetwork()
    assert net.bandwidth == 1.0
    assert net.connect("peer1") is True
    assert net.bandwidth == 1.0  # Bandwidth unchanged after connect
    # Try to connect the same peer again
    assert net.connect("peer1") is False
    assert net.bandwidth == 1.0  # Bandwidth unchanged after failed connect
    # The peer list should only contain one instance
    assert net.peers == ["peer1"]


def test_disconnect_nonexistent_peer_returns_false():
    net = BaseNetwork()
    net.connect("peer1")
    assert net.bandwidth == 1.0
    # Try to disconnect a peer that was never connected
    assert net.disconnect("peer2") is False
    assert net.bandwidth == 1.0  # Bandwidth unchanged after failed disconnect
    # The original peer is still connected
    assert net.peers == ["peer1"]


def test_congestion_factor_zero_does_not_inflate_latency():
    pd.DynAASim().reset()
    dynaa_sim = pd.DynAASim()
    net = BaseNetwork(max_bandwidth=100, max_latency=1.0, congestion_factor=0.0)
    assert net.bandwidth == 100
    assert net.latency == 1.0  # Base latency is 1.0 second

    # Both sends issued at t=0 with 0 in-flight; both start after exactly 1.0 s
    net.send("sender", "receiver", {"size": 1000})
    net.send("sender", "receiver", {"size": 1000})

    dynaa_sim.run(1.001)
    assert len(net._inflight_transmissions) == 2
    assert dynaa_sim.current_time == pytest.approx(1.001)  # Both transfers started after base latency (1.0s)
    assert net.bandwidth == 0  # Bandwidth fully reserved for two concurrent transfers

    # Let run till the end
    dynaa_sim.run()
    assert len(net._inflight_transmissions) == 0
    assert net.latency == 1.0  # Latency returns to base
    assert net.bandwidth == 100  # Bandwidth restored after all transfers complete


def test_congestion_factor_inflates_latency_under_load():
    pd.DynAASim().reset()
    dynaa_sim = pd.DynAASim()
    # congestion_factor=1.0: effective_latency = base * (1 + 1.0 * N)
    net = BaseNetwork(max_bandwidth=100, max_latency=1.0, congestion_factor=1.0)
    assert net.bandwidth == 100
    assert net.latency == 1.0  # Base latency is 1.0 second

    # First send: 0 in-flight -> effective_latency = 1.0 * (1 + 1.0*0) = 1.0 s
    net.send("sender", "receiver", {"size": 1000})

    dynaa_sim.run(1.001)
    assert len(net._inflight_transmissions) == 1  # first transfer has started
    assert dynaa_sim.current_time == pytest.approx(1.001)  # First transfer starts after base latency (1.0s)
    assert net.bandwidth == 0  # Bandwidth reserved for first transfer

    # Second send: 1 in-flight -> effective_latency = 1.0 * (1 + 1.0*1) = 2.0 s
    # So the second transfer starts at t ≈ 1.001 + 2.0 = 3.001
    net.send("sender", "receiver", {"size": 10})

    # Before the inflated latency elapses, still only one transfer active
    dynaa_sim.run(2.5)
    assert len(net._inflight_transmissions) == 1
    assert dynaa_sim.current_time == pytest.approx(2.5)  # Before inflated latency expires
    assert net.bandwidth == 0  # Still only one transfer using bandwidth

    # After the inflated latency elapses, both transfers are active
    dynaa_sim.run(3.002)
    assert len(net._inflight_transmissions) == 2
    assert dynaa_sim.current_time == pytest.approx(3.002)  # Second transfer started after inflated latency
    assert net.bandwidth == 0  # Bandwidth shared between two transfers

    # Let run till the end
    dynaa_sim.run()
    assert len(net._inflight_transmissions) == 0
    assert net.latency == 1.0  # Latency returns to base
    assert net.bandwidth == 100  # Bandwidth restored after all transfers complete
