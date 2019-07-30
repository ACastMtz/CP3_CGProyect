# Programmcode für Übungsblatt 5

Der Code besteht aus mehreren Dateien. Diese müssen einzeln kompiliert und dann
zu einer ausführbaren Datei verlinkt werden. Dies erfolgt automatisch
mit dem Werkzeug `cmake` und den Informationen in `CMakeLists.txt`.

Mehr Informationen zum Thema mehrere Programmcode-Dateien findet man in
[http://www-com.physik.hu-berlin.de/~bunk/cp3/anhA.pdf](http://www-com.physik.hu-berlin.de/~bunk/cp3/anhA.pdf)

## Kompilieren
Die ausführbare Datei `run-cg` wird erzeugt durch
```
cmake .
make
./run-cg
```
Der Punkt in der ersten Zeile ist wichtig. `cmake` erzeugt aus `CMakeLists.txt`
automatisch ein `Makefile`, das von `make` benutzt wird um alle Dateien zu kompilieren und
zu linken. Die erste Zeile muss nur beim ersten Mal ausgeführt werden.

## Eigene Dateien hinzufügen
Eigene Dateien müssen in `CMakeLists.txt` hinzugefügt werden
```
...
      cg.cu
      cg.h
      meine.cu
      meine.h
   )
...
```
und werden dann automatisch kompiliert und gelinkt.