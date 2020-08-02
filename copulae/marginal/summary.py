from typing import List

from scipy.stats.distributions import rv_frozen

from copulae.copula import BaseCopula
from copulae.copula.summary import SummaryType
from .univariate import get_marginal_detail


class Summary(SummaryType):
    def __init__(self, name: str, dim: int, joint_copula: BaseCopula, marginals: List[rv_frozen]):
        self.name = name
        self.dim = dim
        self.marginals = [get_marginal_detail(m) for m in marginals]
        self.joint_copula = joint_copula

    def _repr_html_(self):
        def create_marginal_div(marginal: dict):
            ps = marginal['parameters']
            return f"""
<div>
    <u>Parameters</u>
    <table>
        <thead>
            <tr>
                <th>Distribution</th>
                {''.join(f"<th>{p}</th>" for p in ps)}
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>{marginal['type']}</td>
                {''.join(f"<td>{p}</td>" for p in ps.values())}
            </tr>
        </tbody>
    </table>
</div>
            """.strip()

        html = f"""
<div>
    <h2>{self.name} Copula Summary</h2>
    <div>{self.name} Copula with {self.dim} dimensions (marginals)</div>
    <hr/>
    <h3>Component Details</h3>
    <div>
        <h4>Marginal Details</h4>
        {''.join(create_marginal_div(m) for m in self.marginals)}
    </div>
    <div>
        <h4>Inner Joint Copula</h4>
        {self.joint_copula.summary()}
    <div>
    <hr/>
    <div>
                
    </div>
</div>
"""
        return html

    def __str__(self):
        def create_marginal_block(marginal: dict):
            params = "\n".join([f"\t{k}: {v}" for k, v in marginal['parameters'].items()])
            return f"""
Dist Type: {marginal['type']}
{params}
""".strip()

        marginal_content = '\n'.join(create_marginal_block(m) for m in self.marginals)

        return f"""
{self.name} Copula Summary
{"=" * 80}
{self.name} Copula with {self.dim} dimensions

Marginal Parameters
-------------------
{marginal_content}

{"=" * 80}

Inner Joint Copula Parameter
----------------------------
{self.joint_copula.summary()}
""".strip()
