class NonSymmetricMatrixError(Exception):
    def __init__(self, message: str = None):
        if message is None:
            message = "Input matrix must be symmetric"
        super().__init__(message)
