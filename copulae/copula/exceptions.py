class NotFittedError(Exception):
    def __init__(self):
        super().__init__("Copula has not been fitted")


class InputDataError(Exception):
    pass
