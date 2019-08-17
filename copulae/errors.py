import inspect


class NotApplicableError(Exception):
    """Raised when a method of a class instance is called but when such class should not call the method"""

    def __init__(self, *args):
        if len(args) == 0:
            stack = inspect.stack()[1]
            _class = stack.frame.f_locals['self'].__class__.__name__
            _method = stack.function
            args = [f"'{_method}' is not defined for {_class}"]

        super(NotApplicableError, self).__init__(*args)
