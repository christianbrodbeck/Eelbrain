
clean:
	rm -rf coverage .coverage

test-coverage:
	rm -rf coverage .coverage
	nosetests --with-coverage --cover-package=eelbrain --cover-html --cover-html-dir=coverage

test:
	nosetests eelbrain
