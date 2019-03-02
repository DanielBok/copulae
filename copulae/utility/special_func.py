__all__ = ['stirling_first', 'stirling_first_all', 'stirling_second', 'stirling_second_all']


def stirling_first(n: int, k: int):
    """
    Computes Stirling number of the first kind

    :param n: int
    :param k: int

    :return: int
        Stirling number of the first kind
    """
    if (type(k), type(n)) != (int, int):
        raise TypeError("<k> and <n> must both be integers")

    if k < 0 or k > n:
        raise ValueError("<k> must be in the range of [0, <n>]")

    if n == 0 or n == k:
        return 1
    if k == 0:
        return 0

    s = [1, *[0] * (k - 1)]
    for i in range(1, n):
        last_row = [*s]
        s[0] = - i * last_row[0]
        for j in range(1, k):
            s[j] = last_row[j - 1] - i * last_row[j]

    return abs(s[-1])


def stirling_first_all(n: int):
    """
    Computes all the Stirling number of the first kind for a given <n>

    :param n: int

    :return: List[int]
        a list of all the Stirling number of the first kind
    """
    return [stirling_first(n, k + 1) for k in range(n)]


def stirling_second(n: int, k: int):
    """
    Computes Stirling number of the second kind

    :param n: int
    :param k: int

    :return: int
        Stirling number of the first kind
    """
    if (type(k), type(n)) != (int, int):
        raise TypeError("<k> and <n> must both be integers")

    if k < 0 or k > n:
        raise ValueError("<k> must be in the range of [0, <n>]")

    # s = np.zeros((n, k)).astype(np.uint64)
    if n == 0 or n == k:
        return 1
    if k == 0:
        return 0

    s = [1, *[0] * (k - 1)]
    for _ in range(1, n):
        last_row = [*s]
        for i in range(1, k):
            s[i] = (i + 1) * last_row[i] + last_row[i - 1]

    return s[-1]


def stirling_second_all(n: int):
    """
   Computes all the Stirling number of the second kind for a given <n>

   :param n: int

   :return: List[int]
       a list of all the Stirling number of the second kind
   """
    return [stirling_second(n, k + 1) for k in range(n)]
