@ECHO OFF
:: Create the Sphinx documentation as HTML and show in web browser

:: remember CWD
pushd %~dp0

:: use the value of the command line variable if available
if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -m sphinx
)
:: always use the same apidoc-call (see `python -m sphinx.apidoc --help` for supported arguments)
set SPHINX_APIDOC=python -m sphinx.apidoc --separate --module-first --force

:: optionally use a different builder than HTML if provided
if "%1" == "" (
	set BUILDER=html
) else (
    set BUILDER="%1"
)

set MODULEPATH=..\audioprotopnet
set SOURCEDIR=..\docs
set BUILDDIR=..\docs\_build
set SPHINXPROJ=AudioProtoPNet

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

:: build the rst-files
%SPHINX_APIDOC% -o %SOURCEDIR% %MODULEPATH%

:: build the html-files
%SPHINXBUILD% -M %BUILDER% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:: show the html-files in default browser
explorer.exe %BUILDDIR%\html\index.html

:: return to previous CWD
popd
