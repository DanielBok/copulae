OUTPUT_DIR := dist

.PHONY: build build-all cli dist docs test

all: bdist


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


build-all: build
	@echo "Converting packages"
	python scripts/conda_convert.py $(OUTPUT_DIR)


cli:
	python setup_cli.py develop


dist:
	python setup.py sdist
	python setup.py bdist_wheel



docs:
	rm -rf docs/build
	$(MAKE) -C docs html


test:
	pytest tests/
