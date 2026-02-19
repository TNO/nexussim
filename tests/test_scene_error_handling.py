import pytest

from nexussim.scene.scene import NexusScene


# --- Compute errors ---
def test_duplicate_compute_id():
    scene = {
        "computes": [
            {"id": "c1", "mem": "1 MB", "cpu_speed": "1 GHz", "cpus": 1},
            {"id": "c1", "mem": "2 MB", "cpu_speed": "2 GHz", "cpus": 2},
        ]
    }
    with pytest.raises(ValueError):
        NexusScene.from_dict(scene)


def test_nonpositive_memory():
    # scene = make_scene({"computes": [{"id": "c1"}]})
    scene = {"computes": [{"id": "c1"}]}
    with pytest.raises(ValueError, match="must have positive memory"):
        NexusScene.from_dict(scene)

    scene = {"computes": [{"id": "c1", "mem": "0 MB"}]}
    with pytest.raises(ValueError, match="must have positive memory"):
        NexusScene.from_dict(scene)

    scene = {"computes": [{"id": "c1", "mem": "-10 MB"}]}
    with pytest.raises(ValueError, match="must have positive memory"):
        NexusScene.from_dict(scene)


def test_nonpositive_cpu_speed():
    scene = {"computes": [{"id": "c1", "cpu_speed": "0 Hz", "mem": "1 MB", "cpus": 1}]}
    with pytest.raises(ValueError, match="must have positive CPU speed"):
        NexusScene.from_dict(scene)


def test_nonpositive_cpus():
    scene = {"computes": [{"id": "c1", "mem": "1 MB", "cpu_speed": "1 GHz"}]}
    with pytest.raises(ValueError, match="must have positive CPU count"):
        NexusScene.from_dict(scene)

    scene = {"computes": [{"id": "c1", "cpus": 0, "mem": "1 MB", "cpu_speed": "1 GHz"}]}
    with pytest.raises(ValueError, match="must have positive CPU count"):
        NexusScene.from_dict(scene)

    scene = {"computes": [{"id": "c1", "cpus": -1, "mem": "1 MB", "cpu_speed": "1 GHz"}]}
    with pytest.raises(ValueError, match="must have positive CPU count"):
        NexusScene.from_dict(scene)


# --- Network errors ---
def test_duplicate_network_id():
    scene = {
        "networks": [
            {"id": "n1", "bandwidth": "1 Gbps", "latency": "1 ms", "peers": ["c1"]},
            {"id": "n1", "bandwidth": "2 Gbps", "latency": "2 ms", "peers": ["c1"]},
        ]
    }
    with pytest.raises(ValueError, match="Duplicate network ID"):
        NexusScene.from_dict(scene)


def test_nonpositive_bandwidth():
    scene = {"networks": [{"id": "n1"}]}
    with pytest.raises(ValueError, match="must have a positive bandwidth"):
        NexusScene.from_dict(scene)

    scene = {"networks": [{"id": "n1", "bandwidth": "0 Gbps"}]}
    with pytest.raises(ValueError, match="must have a positive bandwidth"):
        NexusScene.from_dict(scene)

    scene = {"networks": [{"id": "n1", "bandwidth": "-1 Gbps"}]}
    with pytest.raises(ValueError, match="must have a positive bandwidth"):
        NexusScene.from_dict(scene)


def test_negative_latency():
    scene = {"networks": [{"id": "n1", "latency": "-1 ms", "bandwidth": "1 Gbps"}]}
    with pytest.raises(ValueError, match="negative latency"):
        NexusScene.from_dict(scene)


# --- Software component errors ---
def test_duplicate_software_component_id():
    scene = {
        "software components": [
            {
                "id": "sw1",
                "segments": [
                    {
                        "load": {"cpu_load": 1.0, "memory": "1 KB"},
                        "stops_after": {"cycles": 1000},
                    }
                ],
            },
            {
                "id": "sw1",
                "segments": [
                    {
                        "load": {"cpu_load": 1.0, "memory": "1 KB"},
                        "stops_after": {"cycles": 1000},
                    }
                ],
            },
        ]
    }
    with pytest.raises(ValueError, match="Duplicate software component ID"):
        NexusScene.from_dict(scene)


# --- Segment errors ---
def test_nonpositive_cpu_load():
    scene = {
        "software components": [
            {
                "id": "sw1",
                "segments": [
                    {
                        "load": {"cpu_load": -1.0, "memory": "1 KB"},
                        "stops_after": {"cycles": 1000},
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="cpu_load must not be negative"):
        NexusScene.from_dict(scene)


def test_negative_mem_load():
    scene = {
        "software components": [
            {
                "id": "sw1",
                "segments": [
                    {
                        "load": {"cpu_load": 1.0, "memory": "-1 KB"},
                        "stops_after": {"cycles": 1000},
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="mem_load must not be negative, but got -1024"):
        NexusScene.from_dict(scene)
