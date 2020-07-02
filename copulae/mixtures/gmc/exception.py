class GMCFitMethodError(Exception):
    pass


class GMCParamMismatchError(Exception):
    pass


class GMCNotFittedError(Exception):
    def __init__(self):
        super().__init__("GaussianMixtureCopula has not been fitted or given a (default) parameter yet")
