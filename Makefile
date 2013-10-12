
clean:
	rm -rf coverage .coverage
	
clean-py:
	find . -type f -name *.pyc -delete
	find . -type f -name *.pyo -delete

test-coverage:
	rm -rf coverage .coverage
	nosetests --with-coverage --cover-package=eelbrain --cover-html --cover-html-dir=coverage

test:
	nosetests eelbrain
