import sys

import pytest

from nexussim.memory import BoundedMemory, UnboundedMemory


def test_unbounded_memory_initialization():
    max_mem = 1024
    mem = UnboundedMemory(max_mem)
    assert mem.max_memory() == max_mem
    assert mem.used_memory() == 0
    assert mem.free_memory() == max_mem
    assert UnboundedMemory().max_memory() == sys.maxsize


def test_malloc_and_free():
    mem = UnboundedMemory(1024)
    addr1 = mem.malloc(100)
    addr2 = mem.malloc(200)
    assert addr1 != addr2
    assert addr2 == addr1 + 100
    assert mem.used_memory() == 300
    assert mem.free_memory() == 1024 - 300

    assert mem.free(addr1) is True
    assert mem.used_memory() == 200
    assert mem.free(addr1) is False


def test_zero_size_allocation():
    mem = UnboundedMemory(1024)
    with pytest.raises(ValueError):
        mem.malloc(0)
    with pytest.raises(ValueError):
        mem.malloc(-1)
    assert mem.used_memory() == 0
    mem.malloc(1)
    assert mem.used_memory() == 1


def test_invalid_size():
    mem = UnboundedMemory(1024)
    with pytest.raises(TypeError):
        mem.malloc("100")
    with pytest.raises(ValueError):
        mem.malloc(-1)


def test_bounded_memory_basic_allocation_and_free():
    bm = BoundedMemory(1024)
    # allocate two blocks
    a1 = bm.malloc(100)
    a2 = bm.malloc(200)
    assert a1 != a2
    assert bm.used_memory() == 300
    assert bm.free_memory() == 1024 - 300
    assert bm.high_water == 301

    # free first block and ensure it's gone
    assert bm.free(a1) is True
    assert bm.used_memory() == 200
    assert bm.free_memory() == 1024 - 200
    assert bm.high_water == 301
    assert bm.free(a1) is False

    with pytest.raises(MemoryError):
        bm.malloc(1024)

    assert bm.high_water == 301
    a3 = bm.malloc(1024 - 300)
    assert bm.used_memory() == 924
    assert bm.free_memory() == 100
    assert a3 == 301
    assert bm.high_water == 1025


def test_bounded_memory_out_of_memory():
    bm = BoundedMemory(500)
    bm.malloc(400)
    assert bm.used_memory() == 400
    assert bm.free_memory() == 100
    # only 100 left
    with pytest.raises(MemoryError):
        bm.malloc(200)


def test_bounded_memory_reuse_tail_space():
    bm = BoundedMemory(1000)
    bm.malloc(400)
    a2 = bm.malloc(500)
    # high-water now beyond 900
    assert bm.free(a2) is True
    # freeing tail block should allow next malloc to reuse that tail
    a3 = bm.malloc(200)
    assert a3 == a2


def test_bounded_memory_reuse_gap_space():
    bm = BoundedMemory(1000)
    a1 = bm.malloc(300)
    a2 = bm.malloc(300)
    a3 = bm.malloc(300)
    # high-water now beyond 900
    assert bm.used_memory() == 900
    assert bm.free_memory() == 100
    assert bm.high_water == 901

    # free first block and ensure it's gone
    assert bm.free(a1) is True
    assert bm.used_memory() == 600
    assert bm.free_memory() == 400
    assert bm.high_water == 901

    # Let's try and assign to head
    a4 = bm.malloc(300)
    assert a4 == a1
    assert bm.used_memory() == 900
    assert bm.free_memory() == 100
    assert bm.high_water == 901

    # free second block and ensure it's gone
    assert bm.free(a2) is True
    assert bm.used_memory() == 600
    assert bm.free_memory() == 400
    assert bm.high_water == 901

    # Let's try and assign to second block
    a5 = bm.malloc(300)
    assert a5 == a2
    assert bm.used_memory() == 900
    assert bm.free_memory() == 100
    assert bm.high_water == 901

    # free third block and ensure it's gone
    assert bm.free(a3) is True
    assert bm.used_memory() == 600
    assert bm.free_memory() == 400
    assert bm.high_water == 901

    # Let's try and assign to third block
    a6 = bm.malloc(300)
    assert a6 == a3
    assert bm.used_memory() == 900
    assert bm.free_memory() == 100
    assert bm.high_water == 901

    assert bm.free(a1) is True
    a7 = bm.malloc(100)
    assert a7 == a1
    assert bm.used_memory() == 700
    assert bm.free_memory() == 300
    assert bm.high_water == 901

    a8 = bm.malloc(100)
    print(bm._allocations)
    assert a8 == a1 + 100
    assert bm.used_memory() == 800
    assert bm.free_memory() == 200
    assert bm.high_water == 901


def test_bounded_not_enough_memory():
    bm = BoundedMemory(1000)
    a1 = bm.malloc(400)
    bm.malloc(500)
    # high-water now beyond 900
    assert bm.free(a1) is True
    # freeing head block should not allow
    # next malloc to reuse that head, but should find the gap after a1
    with pytest.raises(MemoryError) as memerr:
        bm.malloc(600)
    assert str(memerr.value) == "Not enough memory available"


def test_bounded_no_suitable_block():
    bm = BoundedMemory(1000)
    a1 = bm.malloc(400)
    bm.malloc(500)
    # high-water now beyond 900
    assert bm.free(a1) is True
    # freeing head block should not allow
    # next malloc to reuse that head, but should find the gap after a1
    with pytest.raises(MemoryError) as memerr:
        bm.malloc(500)
    assert str(memerr.value) == "No suitable contiguous block of memory available"
