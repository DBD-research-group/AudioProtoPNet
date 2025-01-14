@ECHO OFF
:: Upload to nexus.

twine upload --repository-url http://fbimvnrepo-01.iee.fraunhofer.de:8081/repository/python-hosted/ dist/* -u jenkins -p jenkins
