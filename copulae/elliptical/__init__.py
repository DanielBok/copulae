from .gaussian import GaussianCopula
from .student import StudentCopula

NormalCopula = GaussianCopula
TCopula = StudentCopula

__all__ = ['NormalCopula', 'GaussianCopula', 'TCopula', 'StudentCopula']
