__all__ = ['CopulaSetupException']


class CopulaSetupException(Exception):
    """
    Exception raised when copula was set up incorrectly. Usually due to incompatible dimensions
    """

    def __init__(self, message='', *args):
        message = f'Copula was setup correctly. {message}'
        super(CopulaSetupException, self).__init__(message, *args)
