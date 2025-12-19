TEXTS = {
    "pl": {
        "language": "Język",
        "title": "Predykcja anulowania rezerwacji hotelowej",
        "description": (
            "Aplikacja wspiera ocenę ryzyka anulowania rezerwacji hotelowych "
            "oraz pokazuje realny wpływ decyzji opartych na modelu na wynik finansowy hotelu."
        ),

        "tab_pred": "🔮 Predykcja",
        "tab_metrics": "📊 Skuteczność i wartość biznesowa",

        "booking": "Dane rezerwacji",

        "lead_time": "Dni do przyjazdu (lead time)",
        "adults": "Liczba dorosłych",
        "children": "Liczba dzieci",
        "prev_cancel": "Liczba wcześniejszych anulowań",
        "prev_not_cancel": "Poprzednie rezerwacje bez anulowania",
        "requests": "Liczba specjalnych próśb",
        "price": "Średnia cena za noc (€)",
        "month": "Miesiąc przyjazdu",
        "dow": "Dzień tygodnia przyjazdu",

        "calc": "Oblicz ryzyko",

        "high": "Wysokie ryzyko anulowania",
        "low": "Niskie ryzyko anulowania",
        "prob": "Prawdopodobieństwo anulowania",


        "kpi_title": "Kluczowe wskaźniki (KPI)",
        "kpi_risk": "Ryzyko anulowania",
        "kpi_season": "Sezonowość",
        "kpi_segment": "Segment rynku",

        "season_high": "Wysoki sezon",
        "season_low": "Poza sezonem",

        "dow_labels": [
            "Poniedziałek", "Wtorek", "Środa",
            "Czwartek", "Piątek", "Sobota", "Niedziela"
        ],

        "explanation": "Dlaczego model tak ocenił to ryzyko?",
        "exp_lead_time": "📅 Długi czas do przyjazdu statystycznie zwiększa ryzyko anulowania.",
        "exp_prev_cancel": "🔁 Klient wcześniej anulował rezerwacje – to silny sygnał ryzyka.",
        "exp_price": "💰 Wyższa cena zwiększa podatność na zmianę planów.",
        "exp_season": "🌞 Wysoki sezon historycznie wiąże się z większą stabilnością rezerwacji.",
        "exp_neutral": "📊 Brak silnych czynników ryzyka – rezerwacja wygląda stabilnie.",


        "metrics_title": "Skuteczność modelu – interpretacja biznesowa",
        "accuracy": "Accuracy (ogólna trafność)",
        "precision": "Precision (jakość alertów)",
        "recall": "Recall (wykrywalność anulowań)",
        "roc_auc": "ROC AUC (zdolność rozróżniania ryzyka)",
        "F1-score": "F1-score (kompromis precyzji i wykrywalności)",

        "confusion_title": "Macierz pomyłek – co to oznacza dla hotelu",

        "cm_tp": "Poprawnie wykryte anulowania – realna szansa na reakcję",
        "cm_fn": "Anulowania niewykryte – bezpośrednia strata przychodu",
        "cm_fp": "Fałszywe alarmy – koszt rabatów lub niepotrzebnej reakcji",
        "cm_tn": "Poprawnie rozpoznane stabilne rezerwacje",

        "cm_summary": (
            "Model nie wykrył {fn} anulowań (FN), co oznacza potencjalną utratę przychodu. "
            "Jednocześnie wygenerował {fp} fałszywych alarmów (FP), "
            "co jest kosztem operacyjnym, ale pozwala chronić większość przychodów."
        ),

        "segment_map": {
            "Online": "Online",
            "Offline": "Offline",
            "Korporacyjny": "Corporate",
            "Lotniczy": "Aviation",
            "Bezpłatny": "Complementary",
        },

        "business_impact_title": "Wpływ biznesowy tej rezerwacji",
        "nights": "Liczba nocy pobytu",
        "expected_loss": "Oczekiwana strata finansowa",

        "business_recommendation_title": "Rekomendacja biznesowa",

        "rec_high_risk": (
            "🔴 Bardzo wysokie ryzyko anulowania.\n\n"
            "Zalecana zaliczka, brak darmowego anulowania lub aktywna retencja klienta."
        ),

        "rec_high_risk_loss": (
            "🟠 Wysokie ryzyko i istotna strata finansowa.\n\n"
            "Rozważ zaliczkę, rabat za brak anulowania lub kontakt z klientem."
        ),

        "rec_high_risk_only": (
            "🟡 Wysokie ryzyko anulowania.\n\n"
            "Rekomendowany rabat lojalnościowy, przypomnienie mailowe lub elastyczna oferta."
        ),

        "rec_low_risk_high_value": (
            "🟠 Niskie ryzyko, ale wysoka wartość rezerwacji.\n\n"
            "Warto monitorować lub zaoferować upsell (np. parking, śniadanie)."
        ),

        "rec_safe": (
            "🟢 Niskie ryzyko anulowania.\n\n"
            "Brak konieczności interwencji – rezerwacja stabilna."
        ),

        "business_value_title": "Wartość biznesowa modelu",
        "business_value_desc": (
            "Poniższe estymacje pokazują, jak model może realnie wpłynąć na wynik finansowy hotelu "
            "przy założeniu aktywnej reakcji na alerty."
        ),

        "business_assumptions_title": "Założenia symulacji",
        "business_assumption_1": "Hotel reaguje na 50% alertów (konserwatywne założenie).",
        "business_assumption_2": "Wartość rezerwacji = liczba nocy × cena za noc.",
        "business_assumption_3": "Skuteczność wykrywania anulowań odpowiada recall modelu.",

        "business_savings_label": "Szacowany zysk na 1 rezerwacji",
        "business_savings_month": "Szacowany zysk miesięczny (100 rezerwacji)",
        "business_value_note": (
            "Wartości są estymacją opartą na danych historycznych i konserwatywnych założeniach. "
            "Rzeczywiste korzyści mogą być wyższe przy lepszej strategii retencji."
        ),
        "months": [
            "Styczeń", "Luty", "Marzec", "Kwiecień",
            "Maj", "Czerwiec", "Lipiec", "Sierpień",
            "Wrzesień", "Październik", "Listopad", "Grudzień"
        ],
        "segment_help": (
            "Segment rynku określa źródło rezerwacji i typ klienta "
            "(np. online, korporacyjny, lotniczy). "
            "Różne segmenty historycznie wykazują różne zachowania anulowań, "
            "dlatego model uwzględnia tę informację przy ocenie ryzyka."
        ),




    },

    "en": {
        "language": "Language",
        "title": "Hotel Booking Cancellation Prediction",
        "description": (
            "The application assesses cancellation risk and demonstrates "
            "the real financial impact of data-driven decisions for hotels."
        ),

        "tab_pred": "🔮 Prediction",
        "tab_metrics": "📊 Model performance & business value",

        "booking": "Booking details",

        "lead_time": "Days before arrival (lead time)",
        "adults": "Number of adults",
        "children": "Number of children",
        "prev_cancel": "Previous cancellations",
        "prev_not_cancel": "Previous non-cancelled bookings",
        "requests": "Special requests",
        "price": "Average price per night (€)",
        "month": "Arrival month",
        "dow": "Arrival weekday",

        "calc": "Calculate risk",

        "high": "High cancellation risk",
        "low": "Low cancellation risk",
        "prob": "Cancellation probability",

        "kpi_title": "Key performance indicators (KPI)",
        "kpi_risk": "Cancellation risk",
        "kpi_season": "Seasonality",
        "kpi_segment": "Market segment",

        "season_high": "High season",
        "season_low": "Low season",

        "dow_labels": [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ],

        "explanation": "Why did the model assess this risk?",
        "exp_lead_time": "📅 Longer lead time statistically increases cancellation risk.",
        "exp_prev_cancel": "🔁 The customer has a history of cancellations.",
        "exp_price": "💰 Higher prices increase sensitivity to plan changes.",
        "exp_season": "🌞 High-season bookings are historically more stable.",
        "exp_neutral": "📊 No strong risk factors detected – booking appears stable.",

        "metrics_title": "Model performance – business interpretation",
        "accuracy": "Accuracy (overall correctness)",
        "precision": "Precision (alert quality)",
        "recall": "Recall (missed revenue protection)",
        "roc_auc": "ROC AUC (risk separation ability)",
        "F1-score": "F1-score (balance between precision and recall)",

        "confusion_title": "Confusion matrix – business meaning",

        "cm_tp": "Correctly detected cancellations",
        "cm_fn": "Missed cancellations – direct revenue loss",
        "cm_fp": "False alarms – operational or discount cost",
        "cm_tn": "Correctly identified stable bookings",

        "cm_summary": (
            "The model missed {fn} cancellations (FN), representing potential lost revenue. "
            "At the same time, it generated {fp} false alerts (FP), "
            "which is an operational trade-off to protect income."
        ),

        "segment_map": {
            "Online": "Online",
            "Offline": "Offline",
            "Corporate": "Corporate",
            "Aviation": "Aviation",
            "Complementary": "Complementary",
        },

        "business_impact_title": "Business impact of this booking",
        "nights": "Number of nights",
        "expected_loss": "Expected financial loss",

        "business_recommendation_title": "Business recommendation",

        "rec_high_risk": (
            "🔴 Very high cancellation risk.\n\n"
            "Deposit, no free cancellation or proactive retention is recommended."
        ),

        "rec_high_risk_loss": (
            "🟠 High cancellation risk and significant potential loss.\n\n"
            "Consider deposit, discount for non-cancellation or customer contact."
        ),

        "rec_high_risk_only": (
            "🟡 High cancellation risk.\n\n"
            "Consider discount, reminder email or flexible retention offer."
        ),

        "rec_low_risk_high_value": (
            "🟠 Low risk but high booking value.\n\n"
            "Monitoring or upsell opportunities recommended."
        ),

        "rec_safe": (
            "🟢 Low cancellation risk.\n\n"
            "No action required – booking appears stable."
        ),

        "business_value_title": "Model business value",
        "business_value_desc": (
            "The estimates below illustrate how the model can improve hotel revenue "
            "through early risk detection and proactive response."
        ),

        "business_assumptions_title": "Simulation assumptions",
        "business_assumption_1": "Hotel reacts effectively to 50% of alerts.",
        "business_assumption_2": "Booking value = number of nights × price per night.",
        "business_assumption_3": "Cancellation detection effectiveness equals model recall.",

        "business_savings_label": "Estimated saving per booking",
        "business_savings_month": "Estimated monthly saving (100 bookings)",
        "business_value_note": (
            "Values are estimates based on historical data and conservative assumptions. "
            "Actual benefits may be higher with optimized retention strategies."
        ),

        "months": [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ],

        "segment_help": (
            "Market segment describes the source and type of booking "
            "(e.g. online, corporate, aviation). "
            "Different segments historically show different cancellation behavior, "
            "which helps the model assess risk more accurately."
        ),
        "f1": "F1-score (precision–recall balance)",

        "f1_desc": (
            "F1-score combines precision and recall, reflecting overall model quality "
            "on imbalanced data. "
            "A slightly lower F1-score is acceptable because the model is intentionally "
            "optimized for higher recall to protect hotel revenue, "
            "even at the cost of more false alerts."
        ),
    }
}