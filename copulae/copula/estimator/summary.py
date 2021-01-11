from itertools import zip_longest
from typing import Optional

import numpy as np

from copulae.copula.summary import SummaryType


class FitSummary(SummaryType):
    """
    Statistics on the fit of the copula

    Attributes
    ----------
    params: named tuple, numpy array
        parameters of the copula after fitting

    method: str
        method used to fit the copula

    log_lik: float
        log likelihood value of the copula after fitting

    nsample: int
        number of data points used to fit copula

    setup: dict
        optimizer set up options

    results: dict
        optimization results
    """

    def __init__(self, params: np.ndarray, method: str, log_lik: float, nsample: int, setup: Optional[dict] = None,
                 results: Optional[dict] = None):
        self.params = params
        self.method = method
        self.log_lik = log_lik
        self.nsample = nsample
        self.setup = setup
        self.results = results

    def as_html(self):
        return self._repr_html_()

    def _repr_html_(self):
        portions = []
        for s, r in zip_longest(self.setup.items(), self.results.items()):
            (sk, sv) = (None, None) if s is None else s
            (rk, rv) = (None, None) if r is None else r
            portions.append('<tr>' + ''.join(f"<td>{i}</td>" for i in (sk, sv, rk, rv)) + '</tr>')

        html = f"""
<div>
    <h3>Fit Summary</h2>
    <hr/>
    <table>
        <tr><th colspan="2">Fit Summary</th></tr>
        <tr><td>Log Likelihood</td><td>{self.log_lik}</td></tr>
        <tr><td>Method</td><td>{self.method}</td></tr>
        <tr><td>Data Points</td><td>{self.nsample}</td></tr>
    </table>
    <br/>
    <table>
        <tr><th colspan="2">Optimization Setup</th><th colspan="2">Results</th></tr>
        {''.join(portions)}
    </table>
</div>
        """
        return html

    def __str__(self):
        msg = [
            f"{'Fit Summary':^80s}",
            '=' * 80,
            f"""
Log. Likelihood      : {self.log_lik}
Method               : {self.method}
Data Points          : {self.nsample}
""".strip(), '']

        skip_keys = {'final_simplex'}
        for title, dic in [('Optimization Setup', self.setup), ('Results', self.results)]:
            if dic is not None:
                string = "\n".join(f'\t{k:15s}: {v}' for k, v in dic.items() if k not in skip_keys)
                msg.extend([title, '-' * 80, string, ''])

        return '\n'.join(msg)
