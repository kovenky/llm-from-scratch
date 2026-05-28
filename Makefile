TEST_PATH=./tests

.PHONY: venv dependencies test-dependencies doc-dependencies clean-venv clean-pyc

.DEFAULT: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

venv: ## Create a virtual environment
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade setuptools
	venv/bin/pip install --upgrade wheel

dependencies: ## Installs packages in requirements.txt into the virtual environment
	pip install -r requirements.txt --no-cache-dir 

test-dependencies: ## Installs packages in test_requirements.txt into the virtual environment
	pip install -r test_requirements.txt --no-cache-dir 

doc-dependencies: dependencies ## Installs packages in doc_requirements.txt into the virtual environment
	pip install -r doc_requirements.txt --no-cache-dir 

update-dependencies:  ## Updates all of the dependency files to the latest versions
	pip-compile requirements.in > requirements.txt	--pip-args "--no-cache-dir"
	pip-compile dev_requirements.in > dev_requirements.txt	--pip-args "--no-cache-dir"
	pip-compile test_requirements.in > test_requirements.txt	--pip-args "--no-cache-dir"
	pip-compile doc_requirements.in > doc_requirements.txt	--pip-args "--no-cache-dir"

clean-venv: ## Uninstall all packages in virtual environment.
	pip freeze | grep -v "^-e" | xargs pip uninstall -y

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +

build-docs: ## Build the html documentation.
	mkdocs build

view-docs: ## Start a web browser pointed at the html documentation.
	open site/index.html

clean-docs: ## Delete all files in the /docs/build directory.
	rm -rf site

unit-tests: clean-pyc  ## Run all tests found in the /tests/unit_tests directory.
	py.test -s --verbose --color=yes $(TEST_PATH)/unit

unit-test-reports: clean-pyc ## Run all tests found in the /tests/unit_tests directory and output unit test and code coverage reports
	# creating the directory to hold unit test report, if it doesn't exist
	mkdir -p sonar_reports
	# running the unit test suite
	# using the python interpreter to execute pytest command so that it will work in CircleCI
	-python -m coverage run --source portfolio_intel -m pytest --junitxml=sonar_reports/unit_tests.xml --verbose --color=yes $(TEST_PATH)/unit
	# creating an xml coverage report
	-python -m coverage xml -o ./sonar_reports/test_coverage.xml
	# creating an html coverage report
	-python -m coverage html -d ./sonar_reports/html_report && zip -r ./sonar_reports/coverage_report.zip ./sonar_reports/html_report && rm -rf ./sonar_reports/html_report
	# deleting the original file
	-rm -rf .coverage

integration-tests: clean-pyc  ## Run all tests found in the /tests/unit_tests directory.
	py.test --verbose --color=yes $(TEST_PATH)/integration

clean-test: ## Delete the test reports
	rm -rf sonar_reports
	rm -rf .pytest_cache


check-codestyle:  ##  Check the style of the code
	pycodestyle portfolio_intel --max-line-length=120

check-docstyle:  ##  Check the style of the docstrings
	pydocstyle portfolio_intel --ignore=D406,D407,D204,D203,D213

security-report:  ## checks for common security vulnerabilities and saves report to file
	mkdir -p sonar_reports
	-bandit -r portfolio_intel --format json > sonar_reports/bandit_report.json

check-security:  ## checks for common security vulnerabilities
	bandit -r portfolio_intel
	