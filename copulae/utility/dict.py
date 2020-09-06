__all__ = ['merge_dict', 'merge_dicts']


def merge_dict(a: dict, b: dict) -> dict:
    """
    Merge 2 dictionaries.

    If the parent and child shares a similar key and the value of that key is a dictionary, the key will be recursively
    merged. Otherwise, the child value will override the parent value.

    Parameters
    ----------
    a dict:
        Parent dictionary

    b dict:
        Child dictionary

    Returns
    -------
    dict
        Merged dictionary
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

    Internally, it calls :code:`merge_dict` recursively.

    Parameters
    ----------
    dicts
        a list of dictionaries

    Returns
    -------
    dict
        Merged dictionary

    See Also
    --------
    :code:`merge_dict`: merge 2 dictionaries
    """
    """

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
