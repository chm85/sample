
def chunks(lst: list, n: int) -> list:
    """
    Converts a list into chunks of smaller lists.

    Parameters
    ----------
    lst: list
         Input list
    n: int
        The number of bins
    Returns
    -------
    chunks: list
        A list of lists.
    """

    chunks = [lst[x:x + n] for x in range(0, len(lst), n)]

    return chunks
