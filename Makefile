
clean:
	rm -rf coverage .coverage dist build
	
clean-py:
	find . -type f -name *.pyc -delete
	find . -type f -name *.pyo -delete

test-coverage:
	rm -rf coverage .coverage
	nosetests --with-coverage --cover-package=eelbrain --cover-html --cover-html-dir=coverage

test:
	nosetests eelbrain

pypi:
	rm -rf build dist doc/build doc/source/generated
	python setup.py sdist upload
	python setup.py build_sphinx
	python setup.py upload_docs
