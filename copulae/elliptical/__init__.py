from .gaussian import GaussianCopula
from .student import StudentCopula
from .abstract import AbstractEllipticalCopula


class NormalCopula(GaussianCopula):
    pass


class TCopula(StudentCopula):
    pass


__all__ = ['AbstractEllipticalCopula', 'NormalCopula', 'GaussianCopula', 'TCopula', 'StudentCopula']
