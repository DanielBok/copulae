CURDIR := $(shell pwd)

.PHONY: clean dist ext wheel linux test

all: dist


dist: wheel clean
	poetry run python setup.py sdist


wheel: clean ext
	poetry run python setup.py bdist_wheel


ext:
	poetry run python setup.py build_ext --inplace


test:
	poetry run python -m pytest tests/


clean:
	rm -rf build/ .pytest_cache/ *.egg-info dist/ __pycache__/ dist/

	# delete cython linker files
	poetry run python .github/utils/find.py remove -pattern *.pyd

	# delete pytest coverage file
	poetry run python .github/utils/find.py remove -pattern *.coverage

linux:
	rm -rf dist/*
	docker container run --rm -v $(CURDIR):/copulae danielbok/manylinux1 /copulae/manylinux-build.sh

write-req:
	poetry export -f requirements.txt --output requirements/build.txt --only=build --without-hashes
	poetry export -f requirements.txt --output requirements/test.txt --with=test,build --without-hashes
