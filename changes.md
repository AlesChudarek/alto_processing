# Změna 1 – adaptivní dělení bloků

- Úprava `main_processor.py` zavádí adaptivní prahy pro dělení textových bloků podle skutečných rozměrů řádků v ALTO.
- Vertikální mezery se nyní porovnávají s prahovou hodnotou odvozenou z mediánu výšek řádků a mediánu zjištěných mezer, což pomáhá držet pohromadě odstavce napříč různými DPI.
- Horizontální posuny mezi řádky se přepočítávají relativně k mediánu šířek řádků a typickým posunům; tím se lépe oddělují vícesloupcové oblasti bez zbytečného štěpení běžných odstavců.
- Zachována je původní logika pro detekci nadpisů (`<h3>`) i vykreslování bloků, takže změna se soustředí jen na spolehlivější rozhodnutí, kdy začít nový blok.
- Heuristické multiplikátory (např. `VERTICAL_GAP_MULTIPLIER`, `HORIZONTAL_WIDTH_RATIO`) jsou nyní soustředěné na začátku `main_processor.py`, takže lze ladit citlivost dělení na jednom místě.
- Při výrazném návratu řádku k levému okraji se dosavadní řádky nově zpětně rozdělí na samostatné bloky; poslední řádek zůstává otevřený pro pokračující odstavec (řeší dialogy a odsazené pasáže). Na stránkách dokumentu `uuid:cfee8750-e69d-11e8-9445-5ef3fc9bb22f` je vidět, že dialogy jsou nyní oddělené jako jednotlivé odstavce a delší věta zůstává spojená s řádkem, kde se text vrací vlevo.
- Upravené konstanty (`HORIZONTAL_WIDTH_RATIO = 0.012`, `HORIZONTAL_SHIFT_MULTIPLIER = 0.85`, `NEGATIVE_SHIFT_MULTIPLIER = 0.85`) snižují práh pro horizontální dělení, takže i dokument `uuid:74317ff3-d7e3-4051-80ec-060e0b6198e2` nepadá do jediného odstavce a respektuje odsazení.
