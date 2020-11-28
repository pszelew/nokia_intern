# Analiza pliku tekstowego *Cell_Phones_&_Accessories.txt*

Badany tekstowy pobrany został ze strony http://snap.stanford.edu/data/web-Amazon-links.html. Na podstawie danych pochodzących z tego pliku tekstowego stworzono proste generatory ocen oraz podsumowania na podstawie tekstu recenzji. Mam nadzieję że to co zrobiłem jest czymś czego oczekiwano :)

## Jak to działa?

1 Sieć neuronowa wytrenowana na 64.000 recenzji próbuje przewidzieć ocenę jaką przypisał/przypisze jej użytkownik. Sprawdzono działanie kilku konfiguracji sieci neuronowej.

2 Sieć neuronowa wytrenowana na 266.336 zestawów: (tekst recenzji) + (słowa znajdujące się już w opisie recenzji) próbuje przewidzieć następne słowo krótkiego podsumowania recenzji. Otrzymywane rezultaty są bardzo ciekawe.

## Więcej szczegółów?

Zapraszam do zapoznania się z zawartością notatników: *01_przewidywanie_ocen.ipynb* oraz *02_generator_tytulow.ipynb*

## Autorzy

* **Patryk Szelewski** - [GitHub](https://github.com/pszelew)
