import random
from collections.abc import Callable


def constant_sampler(load: float = 1.0) -> Callable[[], float]:
    """
    Generates a constant load sampler.

    Works as if sampling from a constant signal.  The reason for this sampler is to be
    able to pass a constant load to methods that recieve a sampler.

    Returns a closure that always returns the specified constant load value.

    Args:
        load (float, optional): The constant  load value. Defaults to 1.0.

    Returns:
        Callable: A function that always returns the specified load value.
    """

    def _sampler():
        return load

    return _sampler


def gaussian_load_sampler(
    min_load: float = 0.0, max_load: float = 2.0, mean: float = 1.0, std: float = 0.5
) -> Callable[[], float]:
    """
     Generates a load sampler using a Gaussian distribution.

    Returns a closure that samples load values from a Gaussian distribution
    characterized by the given mean and standard deviation. The sampled  load is
    constrained to lie within the specified minimum and maximum bounds.

    Args:
          min_load (float, optional): The minimum load allowed. Defaults to 0.0.
          max_load (float, optional): The maximum load allowed. Defaults to 2.0.
          mean (float, optional): The mean of the Gaussian distribution. Defaults to
          1.0.
          std (float, optional): The standard deviation of the Gaussian
          distribution. Defaults to 0.5.

    Returns:
        Callable: A function that samples and returns a constrained  load value.
    """

    def _sampler():
        load = random.gauss(mu=mean, sigma=std)
        if load < min_load:
            return min_load
        if load > max_load:
            return max_load
        return load

    return _sampler


def uniform_load_sampler(min_load: float = 0.0, max_load: float = 2.0) -> Callable[[], float]:
    """
     Generates a  load sampler using a uniform distribution.

    Returns a closure that samples  load values from a uniform distribution
    characterized by the given minimum and maximum bounds.

    Args:
          min_load (float, optional): The minimum  load allowed. Defaults to 0.0.
          max_load (float, optional): The maximum  load allowed. Defaults to 2.0.

    Returns:
        Callable: A function that samples and returns a constrained  load value.
    """

    def _sampler():
        load = random.uniform(min_load, max_load)
        return load

    return _sampler


def exponential_load_sampler(mean: float = 1.0, max_load: float = 10.0) -> Callable[[], float]:
    """
     Generates a  load sampler using an exponential distribution.

    Returns a closure that samples  load values from an exponential distribution
    characterized by the given minimum and maximum bounds.

    Args:
          mean (float, optional): The mean of the exponential distribution. Defaults to
          1.0. max_load (float, optional): The maximum  load allowed. Defaults to 10.0.

    Returns:
        Callable[[], float]: A function that samples and returns a constrained  load
        value.

    Sees Also:
        https://docs.python.org/3/library/random.html
    """

    def _sampler() -> float:
        if mean <= 0.0:
            raise ValueError("Mean must be greater than 0.0")

        load = random.expovariate(1.0 / (mean))
        if load > max_load:
            return max_load
        return load

    return _sampler
