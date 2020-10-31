from copulae.utility.dict import merge_dicts


def test_merge_dicts():
    d1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
    d2 = {'e': 5, 'b': {'c': 3, 'f': 4}}

    final = merge_dicts(d1, d2)
    expected = {'a': 1, 'b': {'c': 3, 'd': 3, 'f': 4}, 'e': 5}

    assert final == expected
    assert merge_dicts(d1) == d1
