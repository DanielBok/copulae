class GMCFitMethodError(Exception):
    pass


class GMCParamMismatchError(Exception):
    pass


class GMCParamError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class GMCNotFittedError(Exception):
    def __init__(self):
        super().__init__("GaussianMixtureCopula has not been fitted or given a (default) parameter yet")
