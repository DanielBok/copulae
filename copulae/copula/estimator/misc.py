def warn_no_convergence():
    print("Warning: Possible convergence problem with copula fitting")


def is_elliptical(copula):
    return copula.name.lower() in ('gaussian', 'student')


def is_archimedean(copula):
    return copula.name.lower() in ('clayton', 'gumbel', 'frank', 'joe', 'amh')
