# AI-podržani digitalni blizanac za prediktivno održavanje

Ovaj repozitorijum sadrži kod razvijen u okviru master rada koji se bavi razvojem AI-podržanog digitalnog blizanca za procenu rizika od otkaza i podršku prediktivnom održavanju u industrijskim IoT sistemima.

U projektu je korišćen AI4I 2020 Predictive Maintenance Dataset, a upoređeni su sledeći modeli:

- Multilayer Perceptron (MLP)
- Random Forest
- XGBoost

MLP model je korišćen kao glavna komponenta dubokog učenja u prototipu digitalnog blizanca.

## Struktura projekta

`ai4i_model_experiments.py`

Sadrži pripremu i pretprocesiranje podataka, obučavanje modela, evaluaciju, izbor praga klasifikacije, poređenje modela i petostruku stratifikovanu unakrsnu validaciju.

`digital_twin_prototype.py`

Sadrži softverski prototip AI-podržanog digitalnog blizanca. Prototip učitava prethodno obučen MLP model i simulira obradu operativnih podataka o stanju mašine.

Za svaki ulazni zapis prototip:

- ažurira virtuelno stanje mašine
- procenjuje rizik od otkaza
- klasifikuje nivo rizika
- generiše preporuku za održavanje
- čuva istoriju stanja i upozorenja

## Skup podataka

Korišćen je AI4I 2020 Predictive Maintenance Dataset dostupan preko UCI Machine Learning Repository.

Ciljna promenljiva je `Machine failure`.

Kao ulazne karakteristike koriste se:

- tip proizvoda
- temperatura vazduha
- temperatura procesa
- brzina rotacije
- obrtni moment
- habanje alata

Oznake pojedinačnih tipova otkaza nisu korišćene kao ulazne karakteristike kako bi se izbeglo curenje informacija iz ciljne promenljive.

## Zahtevi

Python 3.x

Glavne biblioteke:

- pandas
- numpy
- scikit-learn
- TensorFlow / Keras
- XGBoost
- matplotlib

## Napomena

AI4I 2020 je sintetički skup podataka. Zbog toga razvijeni digitalni blizanac predstavlja proof-of-concept softverski prototip, a ne potpuno implementiran industrijski digitalni blizanac povezan sa fizičkom mašinom.

Simulirani replay podataka koristi se za demonstraciju načina obrade operativnih podataka u IoT okruženju i ne predstavlja stvarnu vremensku istoriju jedne fizičke mašine.
