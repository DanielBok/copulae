from inspect import Parameter, signature
from itertools import zip_longest
from typing import Dict, Optional, TypedDict, Union

from scipy import stats
from scipy.stats.distributions import rv_frozen

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

__all__ = ["DistDetail", "create_univariate", "get_marginal_detail"]


class DistDetail(TypedDict, total=False):
    type: Union[stats.rv_continuous, str]
    parameters: NotRequired[Dict[str, float]]


def create_univariate(details: DistDetail) -> rv_frozen:
    dist = get_marginal_class(details)
    parameters = details.get('parameters', {})
    for p in signature(getattr(dist, "_parse_args")).parameters.values():
        if p.default is Parameter.empty and p.name not in parameters:
            parameters[p.name] = 0.5

    return dist(**parameters)


def get_marginal_class(details: DistDetail) -> stats.rv_continuous:
    assert "type" in details, "'type' is a required key in the distribution details"
    dist = details["type"]

    if isinstance(dist, rv_frozen):
        raise TypeError("Do not pass in a actualized marginal. Instead pass in the marginal class itself. \n"
                        "i.e., pass in `stats.norm` instead of `stats.norm()`")

    if isinstance(dist, stats.rv_continuous):
        return dist

    if isinstance(dist, str):
        dist = dist.lower().strip()
        if dist in ('normal', 'gaussian'):
            dist = 'norm'
        elif dist == 'student':
            dist = 't'
        elif dist == 'exp':
            dist = 'expon'

        if hasattr(stats, dist):
            dist = getattr(stats, dist)
            if isinstance(dist, stats.rv_continuous):
                return dist

    raise TypeError(f"Invalid distribution type '{details['type']}'")


def get_marginal_detail(marginal: rv_frozen) -> DistDetail:
    dist = marginal.dist  # distribution class type
    dist_type = dist.__class__.__name__.replace("_gen", "")

    params = signature(getattr(dist, "_parse_args")).parameters
    parameters = {}
    for value, (n, p) in zip_longest(marginal.args, params.items()):  # type: Optional[float], (str, Parameter)
        if value is not None:
            parameters[n] = value
        elif n in marginal.kwds:
            parameters[n] = marginal.kwds[n]
        elif p.default == Parameter.empty:
            parameters[n] = p.default
        else:
            parameters[n] = p.default

    return {
        "type": dist_type,
        "parameters": parameters
    }
