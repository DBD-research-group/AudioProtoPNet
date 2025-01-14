
===========================
 AudioProtoPNet README
===========================


Das ist die *README*-Datei für das Projekt *AudioProtoPNet*.

Diese Datei sollte UTF-8 kodiert sein und nutzt die Formatierungssprache
`reStructuredText <http://docutils.sourceforge.net/rst.html>`_.
Diese Readme wird potentiel genutzt um die Projekt-Webseite zu erstellen,
oder auch als Einstieg zu einem gitlab-Repository. Sie richtet
sich also vornehmlich an Entwickler und soll einen ersten Einstieg geben.

Hauptsächlich wird sie aber von Entwicklern direkt im Texteditor gelesen werden.
Sie ersetzt nicht die Dokumentation die sich im ``docs/`` Verzeichnis befindet.

Typischerweise enthält diese Datei eine Übersicht darüber wie eine Umgebung
für das Projekt einzurichten ist, einige Beispiele zur Nutzung der Build-Tools, usw.
Die Liste der Änderungen (Changelog) sollte nicht hier liegen (sondern in ``docs/changelog.rst``),
aber ein kurzer *“Aktuelle Änderungen”* Abschnitt für die neueste Version ist angemessen.


Repository
==========

Momentan ist als VCS für *AudioProtoPNet* das Fraunhofer Gitlab vorgesehen.
Um eine reibungslose GIT Integration zu gewährleisten gibt es das File ``.gitignore``, wo aufgelistet
wird, welche Dateien nicht mit eingecheckt werden sollen. Das sind im
Allgemeinen alle automatisch generierten Dateien, also insbesondere
die generierte Dokumentation. Streng genommen könnte man aber
auch ein anderes Versionskontrollsystem (etwa SVN) verwenden.
In diesem Fall sollte man die``.gitignore`` löschen.

Im Ordner ``/tests`` wird die Struktur des Hauptordners
(z.B. Ordner ``AudioProtoPNet/audioprotopnet``) nachgebildet.
Alle Dateien im Ordner ``/tests`` haben das Prefix ``test_``,
also z.B. ``test_audioprotopnet.py``.

Neben der GIT Integration kann durch die ``.gitlab-ci.yml`` Datei automatisch ein Gitlab CI/CD
Pipeline erstellt werden. Diese erstellt die Dokumentation, Testet, Überprüft Pylint und baut das Paket.

Notebooks sollten überhaupt nicht in dieses Repository eingecheckt werden, außer
sie gehören wirklich zum Paket.

Einrichtung & Konfiguration
===========================

**TODO:** Der ganze Abschnitt ist nicht besonders projektspezifisch und sollte im
*Confluence* untergebracht werden. Hier stände dann nur ein Link, bzw. besonderheiten von *Commons*.

Die Konfiguration des Projekts wird größtenteils über die ``setup.py`` vorgenommen.
Ein Beispiel für eine solche Datei findet sich im `Cookiecutter-Template <https://gitlab.cc-asp.fraunhofer.de/iee_oe224/Data_Science_project_template>`.
Üblicherweise müssen im oberen Bereich einige Einstellungen für das jeweilige Projekt angepasst werden.

Um das Modul in der Komandozeile verfügbar zu machen und in Python mittels ``import audioprotopnet``
importieren zu können, muss Python wissen wo es liegt. Eine Anleitung findet sich unter
http://confluence.iwes.fraunhofer.de/display/Organisation/OE224+Python.


Tests
=====

Tests werden mit *PyTest* ausgeführt.
Damit diese laufen muss das Modul ``audioprotopnet``
zum Import verfügbar sein (siehe oben).

Anschließend könnnen Tests mit PyCharm via PyTest ausgeführt werden (Start in der Konsole ist auch möglich).
PyTest sucht selbständig nach Files die entweder mit "test_" starten oder mit "_test" enden.

Sofern *Coverage* installiert ist können die Tests auch inklusive
“Coverage-Report” ausgeführt werden:

    scripts\\test_with_coverage.bat

Dies erzeugt einen HTML-Report, der im (Standard-) Internet Browser angezeigt wird.
In diesem Report kann man direkt sehen welche Code-Teile noch nicht von Tests
abgedeckt sind. Das Ziel sind *100% test coverage*, wenn der Wert also darunter liegt
sollte man noch mehr Tests schreiben.


Dokumentation
=============

Die gesamte verfügbare Funktionalität sollte über Docstrings dokumentiert sein.
Das beinhaltet *Pakete*, *Module*, *Funktionen*, *Klassen* und *(public) Methoden*.
Nicht öffentliche Methoden und Funktionen sollten nach Möglichkeit auch dokumentiert sein
um die spätere Anpassung zu vereinfachen.

Wir verwenden `Sphinx <www.sphinx-doc.org/en/master/>`_ zur Dokumentationserstellung.
Dabei verwenden wir die Sprachanpassungen von
`Napoleon <sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_
mit `Google Style Docstrings <sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
für die Formattierung.
Docstrings müssen also entsprechend formatiert sein um später in der Doku richtig
dargestellt zu werden! In PyCharm können die Google Docstring direkt eingestellt werden.


Coding-Konventionen
===================

Bei Code-Anpassungen ist darauf zu achten, dass die Style-Konventionen eingehalten werden.

**TODO:** Link zu den Code-Konventionen im Confluence.
