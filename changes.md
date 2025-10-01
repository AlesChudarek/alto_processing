# Změna 1 – adaptivní dělení bloků

- Úprava `main_processor.py` zavádí adaptivní prahy pro dělení textových bloků podle skutečných rozměrů řádků v ALTO.
- Vertikální mezery se nyní porovnávají s prahovou hodnotou odvozenou z mediánu výšek řádků a mediánu zjištěných mezer, což pomáhá držet pohromadě odstavce napříč různými DPI.
- Horizontální posuny mezi řádky se přepočítávají relativně k mediánu šířek řádků a typickým posunům; tím se lépe oddělují vícesloupcové oblasti bez zbytečného štěpení běžných odstavců.
- Zachována je původní logika pro detekci nadpisů (`<h3>`) i vykreslování bloků, takže změna se soustředí jen na spolehlivější rozhodnutí, kdy začít nový blok.
- Heuristické multiplikátory (např. `VERTICAL_GAP_MULTIPLIER`, `HORIZONTAL_WIDTH_RATIO`) jsou nyní soustředěné na začátku `main_processor.py`, takže lze ladit citlivost dělení na jednom místě.
- Upravené konstanty (`HORIZONTAL_WIDTH_RATIO = 0.012`, `HORIZONTAL_SHIFT_MULTIPLIER = 0.85`, `NEGATIVE_SHIFT_MULTIPLIER = 0.85`) snižují práh pro horizontální dělení, takže i dokument `uuid:74317ff3-d7e3-4051-80ec-060e0b6198e2` nepadá do jediného odstavce a respektuje odsazení.

# Změna 2 – podpora jednořádkových odstavců

- Při výrazném návratu řádku k levému okraji se dosavadní řádky nově zpětně rozdělí na samostatné bloky; poslední řádek zůstává otevřený pro pokračující odstavec (řeší dialogy a odsazené pasáže). Na stránkách dokumentu `uuid:cfee8750-e69d-11e8-9445-5ef3fc9bb22f` je vidět, že dialogy jsou nyní oddělené jako jednotlivé odstavce a delší věta zůstává spojená s řádkem, kde se text vrací vlevo.

# Změna 3 – rozšířené heuristiky a statistická analýza

- Refaktor `main_processor.py` zavádí pokročilé heuristiky pro dělení bloků, dynamickou detekci nadpisových fontů a statistickou analýzu pro lepší přesnost formátování.
- Adaptivní prahy pro vertikální a horizontální dělení bloků se počítají dynamicky z mediánu výšek řádků, mezer a posunů, což zlepšuje rozpoznání odstavců a sloupců.
- Zavedeny statistiky pro fonty: počítá se zastoupení znaků podle velikosti, tučnosti, kurzívy a rodiny fontu, což umožňuje přesnější označení nadpisů `<h1>`, `<h2>` a `<h3>`.
- Dynamická detekce nadpisových fontů na základě rozdílů ve velikosti písma a jejich relativního zastoupení v textu.
- Vylepšené dělení bloků reaguje na horizontální posuny a rozdíly ve velikosti fontu mezi řádky.
- Přidána podpora distribuovaného vzorkování stránek knihy pro výpočet průměrné výšky písma, což zvyšuje přesnost heuristiky.
- Integrace s kontextem knihy a metadaty pro informovanější rozhodování o formátování.
- Přidány rozsáhlé ladicí výpisy pro sledování prahů a rozhodovacích kroků.
- Celkově robustnější a adaptivnější logika pro různorodé ALTO data, výrazně zlepšující kvalitu převodu.
