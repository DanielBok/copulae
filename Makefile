OUTPUT_DIR := dist

.PHONY: bdist dev dev-r docs build build-all test upload

all: build-all

bdist: clean
	python setup.py sdist
	python setup.py bdist_wheel

clean:
	python scripts/clean.py

dev:
	python setup.py develop

dev-r:
	python setup.py develop --uninstall

docs:
	rm -rf docs/build
	$(MAKE) -C docs html

upload:
	@echo "Uploading packages"
	python scripts/upload.py $(OUTPUT_DIR)

build-all: build
	@echo "Converting packages"
	python scripts/conda_convert.py $(OUTPUT_DIR)

build:
	@echo "When building recipe, make sure your conda environment has conda-build and conda-verify installed"

	rm -rf $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR)

# dash in front of command to continue execution even on error. There's a bug that causes conda build to raise an
# error even though the command works fine. Bug likely due to vc runtime for windows
	-conda build --output-folder $(OUTPUT_DIR) conda.recipe

	@echo "------------------ Build complete. Purging build environment ------------------"
	conda build purge
# remove python build directory and egg fodler
	rm -rf build/ allopy.egg-info/

test:
	pytest --cov=copulae --cov-report=term-missing tests/