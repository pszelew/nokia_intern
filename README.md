# Analiza pliku tekstowego *Cell_Phones_&_Accessories.txt*

Na podstawie danych pochodzących z pliku tekstowego *Cell_Phones_&_Accessories.txt* stworzono proste generatory ocen oraz podsumowania na podstawie tekstu recenzji. Dane pobrane został ze strony http://snap.stanford.edu/data/web-Amazon-links.html.  

Mam nadzieję że rezultaty mojej pracy okażą się ciekawe i że są zbieżne z oczekiwaniami. Do tworzenia sieci neuronowych wykorzystałem pakiet keras. Do manipulacji na danych korzystałem głównie z biblioteki NumPy.

## Jak to działa?

1 Sieć neuronowa wytrenowana na 64.000 recenzji próbuje przewidzieć ocenę jaką przypisał/przypisze jej użytkownik. Sprawdzono działanie kilku konfiguracji sieci neuronowej.

2 Sieć neuronowa wytrenowana na 266.336 zestawów: (tekst recenzji) + (słowa znajdujące się już w opisie recenzji) próbuje przewidzieć następne słowo krótkiego podsumowania recenzji. Otrzymywane rezultaty są bardzo ciekawe.

## Więcej szczegółów?

Zapraszam do zapoznania się z zawartością notatników: *01_przewidywanie_ocen.ipynb* oraz *02_generator_tytulow.ipynb*

## Wykorzystane wersje bibliotek

- Tensorflow: 2.3.1
- NumPy:      1.18.1
- matplotlib: 3.1.3
- pickle:     4.0

## Autorzy

* **Patryk Szelewski** - [GitHub](https://github.com/pszelew)
