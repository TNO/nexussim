import sys
from typing import Protocol


class MemoryProvider(Protocol):
    def max_memory(self) -> int:
        """
        Returns the maximum amount of memory available.

        This method should be implemented to return the total memory capacity that the
        provider can allocate, in bytes.

        Returns:
            int: The maximum memory capacity in bytes.
        """
        raise NotImplementedError

    def free_memory(self) -> int:
        """
        Returns the amount of memory available for allocation.

        This method should be implemented to return the difference between the total
        memory capacity and the currently allocated memory, in bytes.

        Returns:
            int: The free memory in bytes.
        """
        raise NotImplementedError

    def used_memory(self) -> int:
        """
        Returns the amount of memory that is currently allocated.

        This method should be implemented to return the sum of all memory blocks
        allocated by the malloc method, in bytes.

        Returns:
            int: The used memory in bytes.
        """
        raise NotImplementedError

    def malloc(self, size: int, requester: str = None) -> int:
        """
        Allocates a memory block of a given size.

        This method should be implemented to allocate a memory block of the given size,
        in bytes.

        Args:
            size (int): The size of the memory block to allocate, in bytes. requester
            (str, optional): The name of the requester of the memory
                allocation. Defaults to None.

        Returns:
            int: The address of the allocated memory block.

        Raises:
            MemoryError: If the requested memory block size is larger than the
                available free memory.
        """
        raise NotImplementedError

    def free(self, address: int) -> None:
        """
        Frees a previously allocated memory block.

        This method should be implemented to release the memory block associated with
        the provided address, making it available for future allocations.

        Args:
            address (int): The address of the memory block to free.

        Returns:
            None
        """
        raise NotImplementedError


class UnboundedMemory:
    """A memory that, despite having a limited capacity, will not fail ever on memory
    allocation.

    Created to models where we do not want to deal with lack of memory.
    """

    def __init__(self, max_memory: int = sys.maxsize):
        """
        Initializes the UnboundedMemory with a maximum memory capacity.

        Args:
            max_memory (int): The maximum memory capacity that can be allocated, in
            bytes.

        Initializes the used memory to 0.
        """
        # track allocations as address -> size (bytes)
        self._allocations: dict[int, int] = {}
        self._next_address = 1
        # use the provided max_memory (keeps API consistent with other providers)
        self._max_memory = max_memory

    def max_memory(self) -> int:
        """
        Returns:
           int: Return the configured maximum memory (bytes).
        """
        return self._max_memory

    def free_memory(self) -> int:
        """
        Returns:
            int: Remaining memory.
        """
        return self._max_memory - self.used_memory()

    def used_memory(self) -> int:
        """
        Returns:
            int: Total bytes currently allocated..
        """
        return sum(self._allocations.values())

    def malloc(self, size: int, requester: str | None = None) -> int:
        """Allocate a block of memory and return its address.

        This unbounded implementation always succeeds (even if it exceeds
        `max_memory`).

        Args:
            size (int): The size of the memory block to allocate, in bytes. requester
            (str, optional): The name of the requester of the memory
                allocation. Defaults to None.

        Returns:
            int: The address of the allocated memory block.

        Raises:
            MemoryError: If the requested memory block size is larger than the
        """
        if not isinstance(size, int):
            raise TypeError("size must be an int (bytes)")
        if size <= 0:
            raise ValueError("size must be non-negative")

        assigned_address = self._next_address
        self._allocations[assigned_address] = size

        self._next_address += size
        return assigned_address

    def free(self, address: int) -> bool:
        """
        Free a previously allocated block.

        Args:
            address (int): The address the memory is at.

        Returns:
            bool: True if freeing the memory release was successful, False otherwise.
        """
        try:
            self._allocations.pop(address)
            return True
        except KeyError:
            return False


class BoundedMemory:
    """A memory provider that enforces a maximum memory capacity.

    Allocations are tracked as address -> size (bytes). Freed addresses are
    recycled to avoid address space exhaustion in long-running tests.
    """

    def __init__(self, max_memory: int):
        if not isinstance(max_memory, int):
            raise TypeError("max_memory must be an int (bytes)")
        if max_memory < 0:
            raise ValueError("max_memory must be non-negative")

        # track allocations as address -> size (bytes)
        self._allocations: dict[int, int] = {}
        # next address to try when allocating at high-water mark
        self._high_water: int = 1
        self._max_memory = max_memory

    @property
    def high_water(self) -> int:
        """high_water is the next address after the current highest allocated byte.
        This address is not guaranteed to exist!
        If high_water is beyond max_memory, it means that
        there is no more contiguous space at the end of the address space,
        but there may still be free memory in gaps between allocated blocks.
        This means that it is trivial to find the highest address ever in use
        (high_water -1).
        """
        return self._high_water

    def _bump_high_water(self, next_addr: int, size: int) -> None:
        """Ensure the internal high-water mark is at least `next_addr + size`.

        This encapsulates the monotonic update so call sites don't need to
        repeat it repeat it.
        """
        if next_addr + size > self._high_water:
            self._high_water = next_addr + size

    def max_memory(self) -> int:
        return self._max_memory

    def free_memory(self) -> int:
        return self._max_memory - self.used_memory()

    def used_memory(self) -> int:
        return sum(self._allocations.values())

    def malloc(self, size: int, requester: str | None = None) -> int:
        """Allocate a block of memory and return its address.
        Allocation strategy:
        1. If there is enough free memory, try to allocate at the high-water mark
           (addresses beyond any previous allocation). This minimizes fragmentation.
        2. If that doesn't fit, scan for gaps between existing allocations and use
           the first contiguous gap that fits.

        Raises MemoryError if there is not enough free memory or no suitable
        contiguous block is found.
        """
        if not isinstance(size, int):
            raise TypeError("size must be an int (bytes)")
        if size < 0:
            raise ValueError("size must be non-negative")
        if size > self.free_memory():
            raise MemoryError("Not enough memory available")

        # Try high-water allocation first
        if self._high_water + size < self._max_memory:
            next_addr = self._high_water
            self._allocations[next_addr] = size
            self._bump_high_water(next_addr, size)
            return next_addr

        # Build a sorted list of candidate addresses to scan for gaps.
        # Make sure we always start at 1, even if there are no allocations,
        # to check for space at the beginning.
        addresses = sorted({1} | set(self._allocations))

        # Iterate through candidate start addresses in order.
        for i, addr in enumerate(addresses):
            # compute the byte immediately after the current block (0 if no block)
            next_addr = addr + self._allocations.get(addr, 0)

            # Check two conditions together:
            # - Either we are at the last address or it fits before the next address
            # - The request must also fit within the maximum memory bound
            end = next_addr + size
            if (i == len(addresses) - 1 or end <= addresses[i + 1]) and end - 1 <= self._max_memory:
                # Found a gap large enough to fit the requested block; allocate
                # the block at `next_addr`.
                self._allocations[next_addr] = size
                # bump high water mark (if needed)
                self._bump_high_water(next_addr, size)
                return next_addr

        # No contiguous block found
        raise MemoryError("No suitable contiguous block of memory available")

    def free(self, address: int) -> bool:
        return self._allocations.pop(address, None) is not None
