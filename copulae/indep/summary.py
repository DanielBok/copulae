from copulae.copula.summary import SummaryType


class FitSummary(SummaryType):
    def __init__(self, dim: int):
        self.dim = dim

    def _repr_html_(self) -> str:
        return f"""
<div>
    <h2>Independent Copula Summary</h2>
    <table>
        <tr>
            <td>Dimensions</td><td>{self.dim}<td/>
        <tr/>
    </table>
    <div>The IndepCopula does not need fitting</div>
</div>
        """.strip()

    def __str__(self) -> str:
        return '\n'.join([
            "Independent Copula Summary",
            "=" * 80,
            f"{'Dimensions':20} : {self.dim}",
            "-" * 80,
            "The IndepCopula does not need fitting"
        ])
