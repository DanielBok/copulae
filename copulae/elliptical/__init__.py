from .Gaussian import GaussianCopula
from .Student import StudentCopula

NormalCopula = GaussianCopula
TCopula = StudentCopula

__all__ = ['NormalCopula', 'GaussianCopula', 'TCopula', 'StudentCopula']
