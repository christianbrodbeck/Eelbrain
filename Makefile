UNAME := $(shell uname -m)

clean:
	rm -rf coverage .coverage build dist

clean-py:
	find . -type f -name *.pyc -delete
	find . -type f -name *.pyo -delete

doc:
	pythonw setup.py build_sphinx

style-check:
	pydocstyle eelbrain

flake:
	flake8 --count eelbrain examples scripts

test: style-check
	pytest eelbrain

testw:
	pythonw $(shell which pytest) eelbrain

test-no-gui:
	pytest eelbrain --no-gui

test-only-gui:
	pythonw $(shell which pytest) eelbrain/_wxgui eelbrain/plot eelbrain/tests/test_examples.py eelbrain/tests/test_fmtxt.py eelbrain/tests/test_report.py

pypi:
	rm -rf build dist
	python setup.py sdist bdist_wheel bdist_egg
	twine upload --repository-url https://upload.pypi.org/legacy/ -u christianbrodbeck dist/*

.PHONY: clean clean-py doc testw pypi
