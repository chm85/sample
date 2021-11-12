from typing import Union


def lrg_or_not(param1: int, param2: Union[int, float]) -> bool:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    return param1 > param2
