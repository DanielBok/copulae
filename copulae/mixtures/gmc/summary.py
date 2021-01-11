import pandas as pd

from copulae.copula.summary import SummaryType
from .parameter import GMCParam


class Summary(SummaryType):
    def __init__(self, params: GMCParam, fit_details: dict):
        self.name = "Gaussian Mixture Copula"
        self.params = params
        self.fit = fit_details

    def _repr_html_(self):
        params = [f"<strong>{title}</strong>" + pd.DataFrame(values).to_html(header=False, index=False)
                  for title, values in [("Mixture Probability", self.params.prob),
                                        ("Means", self.params.means)]]
        params.append(
            f"<strong>Covariance</strong>" +
            '<br/>'.join(
                f"<div>Margin {i + 1}</div>{pd.DataFrame(c).to_html(header=False, index=False)}"
                for i, c in enumerate(self.params.covs)
            )
        )

        fit_details = ''
        if self.fit['method'] is not None:
            fit_details = f"""
<div>
    <h3>Fit Details</h3>
    <div>Algorithm: {self.fit['method']}</div>
</div>
            """.strip()

        html = f"""
<div>
    <h2>{self.name} Summary</h2>
    <div>{self.name} with {self.params.n_clusters} components and {self.params.n_dim} dimensions</div>
    <hr/>
    <div>
        <h3>Parameters</h3>
        {'<br/>'.join(params)}
    </div>
    {fit_details}
</div>
""".strip()

        return html

    def __str__(self):
        return '\n'.join([
            f"{self.name} Summary",
            "=" * 80,
            str(self.params)
        ])
