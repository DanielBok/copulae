from copulae.copula.summary import SummaryType
from copulae.mixtures.gmc import GMCParam


class FitSummary(SummaryType):
    def __init__(self, best_params: GMCParam, has_converged: bool, method: str, nsample: int, setup: dict = None):
        self.best_params = best_params
        self.has_converged = has_converged
        self.method = method
        self.nsample = nsample
        self.setup = setup

    def as_html(self):
        return self._repr_html_()

    def _repr_html_(self):
        setup_details = ''
        if isinstance(self.setup, dict):
            setup_details = ''.join(f'<tr><td>{s}</td><td>{r}</td></tr>' for s, r in self.setup.items())

        html = f"""
<div>
    <h3>Fit Summary</h2>
    <hr/>
    <table>
        <tr><th colspan="2">Fit Summary</th></tr>
        <tr><td>Method</td><td>{self.method}</td></tr>
        <tr><td>Data Points</td><td>{self.nsample}</td></tr>
        {setup_details}
    </table>
</div>
        """
        return html

    def __str__(self):
        msg = [
            f"{'Fit Summary':^80s}",
            '=' * 80,
            f"""
Method               : {self.method}
Data Points          : {self.nsample}
""".strip()]

        if isinstance(self.setup, dict):
            for s, r in self.setup.items():
                msg.append(f"{s:21}: {r}")

        return '\n'.join(msg)
