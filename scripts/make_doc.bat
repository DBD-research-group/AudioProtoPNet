@ECHO OFF

:: Create the Sphinx documentation

:: remember CWD
pushd %~dp0

:: use the value of the command line variable if available
if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -msphinx
)
set SOURCEDIR=..\docs
set BUILDDIR=..\docs\_build
set SPHINXPROJ=AudioProtoPNet

:: builder argument is required
if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The Sphinx python module was not found. Make sure you have Sphinx installed!
	echo.
	echo.You probably only need to install it via
	echo.  pip install sphinx
	echo.or set your path correctly.
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
:: return to previous CWD
popd
