import os


def generate_meta():
    """
    Generates the meta data functions for each of the copula
    """
    from copulae import ClaytonCopula
    ClaytonCopula(2)


def reset_meta():
    try:
        file = os.path.join(os.path.dirname(__file__), 'data_funcs.p')
        os.remove(file)
    except FileNotFoundError:
        pass

    generate_meta()
