.PHONY: dev dev-r

dev:
	python setup.py develop

dev-r:
	python setup.py develop --uninstall
