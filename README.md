# Hotel Booking Cancellation Prediction

## Opis projektu

Celem projektu jest przewidywanie ryzyka anulowania rezerwacji hotelowej na podstawie danych historycznych oraz pokazanie realnej wartości biznesowej wynikającej z zastosowania modelu predykcyjnego.

Projekt składa się z:
- modelu Machine Learning (Logistic Regression),
- aplikacji webowej w Streamlit,
- interpretacji metryk w kontekście biznesowym.

Aplikacja nie tylko przewiduje prawdopodobieństwo anulowania, ale również:
- estymuje potencjalną stratę finansową,
- sugeruje działania biznesowe (zaliczka, rabat, brak interwencji),
- pokazuje, ile hotel może realnie zyskać, korzystając z modelu.


## Problem biznesowy

Anulowania rezerwacji generują dla hoteli:
- utracony przychód,
- koszty operacyjne,
- problemy z obłożeniem pokoi.

Hotel nie może reagować na wszystkie rezerwacje, dlatego potrzebuje:
- systemu priorytetyzacji ryzyka,
- narzędzia wspierającego decyzje (decision support system).


## Cel modelu

Model odpowiada na pytanie:

> Jakie jest prawdopodobieństwo, że dana rezerwacja zostanie anulowana?

Na tej podstawie hotel może:
- wprowadzić zaliczkę,
- zaproponować rabat za brak anulowania,
- wysłać przypomnienie mailowe,
- pozostawić rezerwację bez ingerencji.


## Dane
Dane pochodzą z publicznego zbioru danych (https://www.kaggle.com/datasets/gauravduttakiit/reservation-cancellation-prediction?select=test___dataset.csv) dotyczącego rezerwacji hotelowych (dataset historyczny, dane anonimowe) i zawierają m.in.:

### Zmienne numeryczne
- `lead_time` – liczba dni do przyjazdu  
- `avg_price_per_room` – średnia cena za noc  
- `no_of_special_requests` – liczba specjalnych próśb  
- `no_of_previous_cancellations`  
- `no_of_previous_bookings_not_canceled`  
- `no_of_adults`, `no_of_children`  
- `arrival_month_num`, `arrival_day_of_week`

### Zmienne kategoryczne
- `market_segment_type` (Online, Offline, Corporate, itp.)

### Zmienna docelowa
- `booking_status`  
  - `1` – anulowana  
  - `0` – zrealizowana  


## Pipeline Machine Learning

Model został zbudowany w postaci Pipeline, aby uniknąć wycieków danych i zapewnić spójność przetwarzania.

### 1. Przetwarzanie danych
- zmienne numeryczne: standaryzacja (StandardScaler)  
- zmienne kategoryczne: One-Hot Encoding

### 2. Model
- Logistic Regression
- `class_weight={0:1, 1:2}`  
  (większa waga dla anulowań – są kosztowniejsze biznesowo)

### 3. Walidacja
- 5-fold cross-validation
- metryka: ROC AUC


## Metryki modelu (finalna wersja)

- Accuracy: ~0.77  
- Precision: ~0.59  
- Recall: ~0.77  
- ROC AUC: ~0.85  

### Interpretacja biznesowa:
- Recall – model wykrywa większość anulowań (ochrona przychodu),
- Precision – część alertów jest fałszywa, co jest akceptowalne kosztowo,
- ROC AUC – dobra zdolność rozróżniania ryzykownych i stabilnych rezerwacji.


## Próg decyzyjny

Zastosowano próg: 0.47

Oznacza to, że:
- model jest nastawiony na wykrywanie ryzyka, a nie tylko „pewne” przypadki,
- strategia jest konserwatywna biznesowo.


## Struktura projektu
<pre>
hotel-booking/
│
├── app/
├── data/
│   ├── processed/
│   └── raw/
├── eda/
├── model/
├── requirements.txt
└── README.md
</pre>

Logika projektu jest rozdzielona:
- app/ – serwowanie modelu w postaci aplikacji webowej,
- data/ – przygotowanie i przetwarzanie danych,
- eda/ – analiza eksploracyjna,
- model/ – trening i ewaluacja modelu,

## Uruchomienie aplikacji

Aplikacja jest dostępna online pod danym linkiem, który należy wkleić do przeglądarki internetowej: https://hotel-bookinggit-nkylerkejgzuvdtbieyxsq.streamlit.app


## Przenośność aplikacji
Aplikacja została zaprojektowana w sposób umożliwiający łatwe przenoszenie i uruchamianie w różnych środowiskach. Wszystkie wymagane zależności zostały jednoznacznie określone w pliku requirements.txt, co pozwala na odtworzenie środowiska uruchomieniowego na dowolnym systemie z zainstalowanym Pythonem.

Kod źródłowy projektu wraz z pełną strukturą katalogów dostępny jest w publicznym repozytorium GitHub:

https://github.com/olaola994/hotel-booking

Repozytorium zawiera komplet plików niezbędnych do uruchomienia aplikacji, w tym kod aplikacji, model Machine Learning, dane oraz dokumentację projektu.

Aby uruchomic aplikację lokalnie na swoim kompuetrze należy:
1. sklonować repozytorium 
2. Otworzyc terminal w kompilatorze (np. PyCharm)
2. Przejść do katalogu hotel-booking
cd hotel-booking
3. Uruchomić komenedy w terminalu 
python -m venv .venv
source .venv/bin/activate  # lub Windows activate
python -m pip install -r requirements.lock.txt
streamlit run app.py

## Experiment tracking z użyciem MLflow

W projekcie zastosowano MLflow do śledzenia eksperymentów uczenia maszynowego (experiment tracking).

MLflow umożliwia:
- rejestrowanie parametrów modelu (np. class_weight, próg decyzyjny),
- zapisywanie metryk jakości modelu (Accuracy, Precision, Recall, F1-score, ROC AUC),
- porównywanie różnych wariantów modelu w sposób uporządkowany,
- przechowywanie wytrenowanych modeli jako artefaktów.

W ramach projektu przeprowadzono kilka eksperymentów:
- baseline – model bazowy bez wag klas, z domyślnym progiem 0.5,
- high_recall – model z większą wagą dla anulowań (class imbalance),
- final_model – model końcowy z dobranym progiem decyzyjnym (0.47), zoptymalizowany pod kątem kompromisu między recall i precision.

Każdy eksperyment jest zapisywany jako osobny run w MLflow, co pozwala jasno wykazać:
- wpływ niezbalansowanych danych na metryki,
- kompromis między wykrywalnością anulowań a liczbą fałszywych alarmów,
- zasadność wyboru finalnego modelu na podstawie metryki Recall.

MLflow uruchamiany jest lokalnie i nie wymaga zewnętrznych usług chmurowych.

Uruchomienie MLflow

<pre>
python -m model.train2
mlflow ui
</pre>