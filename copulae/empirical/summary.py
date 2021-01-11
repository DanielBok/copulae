from copulae.copula.summary import SummaryType


class FitSummary(SummaryType):
    def __init__(self, n: int, dim: int):
        self.n = n
        self.dim = dim

    def _repr_html_(self) -> str:
        return f"""
<div>
    <h2>Empirical Copula Summary</h2>
    <table>
        <tr>
            <td>Dimensions</td><td>{self.dim}</td>
            <td>No. Observations</td><td>{self.n}<td/>
        <tr/>
    </table>
    <div>The EmpiricalCopula does not need fitting</div>
</div>
        """.strip()

    def __str__(self) -> str:
        return '\n'.join([
            "Empirical Copula Summary",
            "=" * 80,
            f"{'Dimensions':20} : {self.dim}",
            f"{'No. Observations':20} : {self.n}",
            "-" * 80,
            "The EmpiricalCopula does not need fitting"
        ])
