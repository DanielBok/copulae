from functools import wraps


def merge_dict(a: dict, b: dict) -> dict:
    """
    Merge 2 dictionaries.

    If the parent and child shares a similar key and the value of that key is a dictionary, the key will be recursively
    merged. Otherwise, the child value will override the parent value.

    :param a: dict
        parent dictionary
    :param b: dict
        child dictionary
    :return: dict
        merged dictionary
    """

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = merge_dict(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries recursively

    :param dicts: List[Dict]
        a list of dictionaries
    :return: dict
        merged dictionary
    """

    a = dicts[0]
    if len(dicts) == 1:
        return dicts[0]

    for b in dicts:
        a = merge_dict(a, b)
    return a


def format_docstring(*args, **kwargs):
    def decorator(func):
        func.__doc__ = func.__doc__.format(*args, **kwargs)
        return func

    return decorator
