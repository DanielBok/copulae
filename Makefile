OUTPUT_DIR := dist
CURDIR := $(shell pwd)

.PHONY: cli dist dist-wheel test

all: dist


dist: wheel clean
	python setup.py sdist


wheel: clean ext
	python setup.py bdist_wheel


ext:
	python setup.py build_ext --inplace


test:
	python -m pytest tests/


clean:
	rm -rf build/ .pytest_cache/ *.egg-info dist/ __pycache__/ dist/

	# delete cython linker files
	find . -type f -name '*.pyd' -delete

	# delete pytest coverage file
	find . -type f -name '*.coverage' -print

linux:
	rm -rf dist/*
	docker container run --rm -v $(CURDIR):/copulae danielbok/manylinux1 /copulae/manylinux-build.sh

conda:
	conda build --output-dist dist conda.recipe
