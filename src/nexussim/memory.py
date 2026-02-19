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
        ...

    def free_memory(self) -> int:
        """
        Returns the amount of memory available for allocation.

        This method should be implemented to return the difference between the total
        memory capacity and the currently allocated memory, in bytes.

        Returns:
            int: The free memory in bytes.
        """
        ...

    def used_memory(self) -> int:
        """
        Returns the amount of memory that is currently allocated.

        This method should be implemented to return the sum of all memory blocks
        allocated by the malloc method, in bytes.

        Returns:
            int: The used memory in bytes.
        """
        ...

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
        ...

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
        ...


class UnboundedMemory:
    """A memory that, despite having a limited capacity, will not fail ever on memory
    allocation.

    Created to models where we do not want to deal with lack of memory.
    """

    def __init__(self, max_memory: int):
        """
        Initializes the UnboundedMemory with a maximum memory capacity.

        Args:
            max_memory (int): The maximum memory capacity that can be allocated, in
            bytes.

        Initializes the used memory to 0.
        """
        self._used_memory = 0
        self._max_memory = max_memory
