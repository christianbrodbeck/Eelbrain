
test-coverage:
	rm -rf coverage .coverage
	nosetests --with-coverage --cover-package=eelbrain --cover-html --cover-html-dir=coverage

clean:
	rm -rf coverage .coverage