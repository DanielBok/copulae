from .gaussian import GaussianCopula
from .student import StudentCopula


class NormalCopula(GaussianCopula):
    pass


class TCopula(StudentCopula):
    pass


__all__ = ['NormalCopula', 'GaussianCopula', 'TCopula', 'StudentCopula']
