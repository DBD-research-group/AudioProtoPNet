@ECHO OFF
:: Run the tests including coverage and opens the report-html in the Browser.

:: save the folder in the temp dictionary
set HtmlPath=%TEMP%\html_cov_commons

python -m nose ^
    --with-coverage ^
    --cover-package=audioprotopnet ^
    --cover-html ^
    --cover-html-dir=%HtmlPath% ^
    %*

:: open in default browser
explorer %HtmlPath%\index.html
