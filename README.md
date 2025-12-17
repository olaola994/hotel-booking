Ocena jakości występów zawodników FPL na podstawie statystyk meczu

O danych
* Zbiór danych: statystyki występów zawodników Premier League.
* Połączenie 3 plików, usunięcie duplikatów.
* Brak braków danych, typy kolumn OK.

O przetwarzaniu
* Filtrowanie: rozpatrujemy tylko mecze, gdzie zawodnik zagrał (Minutes > 0).
* Definicja etykiety:
    * dobry pick (1) = Points >= 6,
    * zły/średni (0) = Points < 6.
* Cechy wejściowe:
    * statystyki: Minutes, Goals, Assists, BPS, expected_*,
    * oraz Team, Pos.

O modelu i testowaniu
* Podział train/test: 80/20, stratified (zachowanie proporcji klas).
* Model: RandomForestClassifier (200 drzew, random_state=42).
* Metryki:
    * accuracy ~0.98,
    * dla klasy „dobry pick”: precision ~0.95, recall ~0.93, f1 ~0.94,
    * confusion matrix pokazuje, że model rzadko myli się na klasie 1.
* Feature importance:
    * najważniejsze cechy: BPS, Goals, expected_goal_involvements, expected_goals, Minutes.

Dlaczego RandomForest?
* dobrze radzi sobie z nieliniowymi zależnościami między statystykami meczu a etykietą
* nie wymaga skomplikowanego skalowania cech
* daje bardzo dobrą jakość przy naszym zbiorze danych
* pozwala nam wyznaczyć feature importance, więc łatwo zinterpretować, które cechy są najważniejsze
