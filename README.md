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

> *Jakie jest prawdopodobieństwo, że dana rezerwacja zostanie anulowana?*

Na tej podstawie hotel może:
- wprowadzić zaliczkę,
- zaproponować rabat za brak anulowania,
- wysłać przypomnienie mailowe,
- pozostawić rezerwację bez ingerencji.


## Dane

Dane pochodzą z historycznych rezerwacji hotelowych i zawierają m.in.:

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
- **Logistic Regression**
- `class_weight={0:1, 1:2}`  
  (większa waga dla anulowań – są kosztowniejsze biznesowo)

### 3. Walidacja
- 5-fold cross-validation
- metryka: ROC AUC


## Metryki modelu (finalna wersja)