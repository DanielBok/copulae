OUTPUT_DIR := dist

.PHONY: cli dist dist-wheel test

all: dist


cli:
	python setup_cli.py develop


dist:
	python setup.py sdist
	python setup.py bdist_wheel


dist-wheel:
	python setup.py bdist_wheel


ext:
	python setup.py build_ext --inplace


test:
	python -m pytest tests/


clean:
	# uninstall cli
	python setup_cli.py develop --uninstall
	rm -rf .pytest_cache/ *.egg-info dist/ .coverage
