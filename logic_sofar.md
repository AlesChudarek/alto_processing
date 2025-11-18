# Aktuální logika zpracování ALTO

Tento dokument shrnuje současnou implementaci v `main_processor.py` a zároveň
vysvětluje, jak se liší od původního přístupu (`kramerius_alto_service.ts`).

## Jak se lišíme od původní TypeScript verze

- **Lepší získávání dat** – původní služba pracovala jen s dodaným ALTO XML.
  Python procesor umí načítat kontext knihy, fallbackovat mezi více API
  základnami, znovu zkoušet při chybách a cacheovat výsledky.
- **Dynamické prahy** – místo pevných čísel (např. `> 40` pro rozdělení) se
  všechny rozhodovací prahy odvozují od mediánů šířek řádků, výšek písma a
  reálného rozptylu dat na stránce.
- **Analýza fontů a nadpisů** – nově počítáme poměry velikostí fontů, umíme
  identifikovat nadpisové fonty s ohledem na jejich zastoupení, a používáme
  word-by-word analýzu výšek, abychom nadpisy nepromovali falešně.
- **Chytřejší segmentace bloků** – sledujeme vertikální mezery, horizontální
  posuny, negativní odsazení, výšku řádků, změny fontu, centrování i
  znovu-centrování. Výsledkem je stabilnější rozdělení na odstavce i u zcela
  rozhozených OCR výstupů.
- **Detekce malého textu** – footnotes/popisky se filtrují stejnou logikou jako
  nadpisy, s možností ulevit prahům pro hvězdičky nebo původní `h3`.
- **Page-number logika** – analyzujeme kandidátní řádky, párujeme je s
  metadata anotacemi, umíme povýšit čistě číselné sekundární kandidáty a
  přidáváme `<note>` anotace do výstupu.
- **Tokenové zpracování** – zachycujeme HPOS jednotlivých tokenů, ignorujeme
  úvodní OCR šum a umíme správně pracovat s `HypPart1/2`, aby se slova
  nezobrazovala dvakrát, ale zároveň se počítala do geometrie.
- **Debug a ladění** – procesor tiskne detailní `DEBUG` logy, uchovává důvody
  splitů (`split_reason`) a poskytuje Utility skripty (`test_final.py`,
  `test_input_api.py`, `batch_comparison_web.py`) pro testování velkých sad
  UUID.

## Architektura `main_processor.py`

### Načítání a příprava dat

- **API vrstva** – `AltoProcessor` drží seznam API základen, normalizuje je a
  po úspěšném dotazu si pamatuje, která základna fungovala naposledy.
- **HTTP session** – využíváme `requests.Session` s retry politikou (429, 5xx),
  aby volání Krameria bylo robustní proti výpadkům.
- **Book context** – metoda `get_book_context` podle UUID zjistí, jestli jde o
  stránku nebo knihu, rekurzivně projde child uzly, stáhne MODS metadata a
  vybere reprezentativní stránky pro statistiku.
- **Výběr vzorků** – stránky načítáme ve vlnách (`TEXT_SAMPLE_WAVE_SIZE`),
  dokud nezískáme dost dat nebo dokud nedosáhneme `MIN_CONFIDENCE_FOR_EARLY_STOP`
  (např. 0.9). Každá vlna se vyhodnocuje podle počtu validních slov.
- **Caching** – statistiky o knize se ukládají do globálního cache, abychom
  při opakovaném zpracování stejné knihy nemuseli všechno počítat znovu.

### Výpočet typografických statistik

- **Textové styly** – z ALTO `TextStyle` elementů extrahujeme velikosti, bold,
  italic, font names a skládáme z nich “podpisy” pro jednotlivá slova.
- **Metriky slov** – pro každý token získáváme výšku, šířku, délku (počet
  ne-mezer), detekujeme bold, uchováváme HPOS/VPOS i reálný obsah (po
  odstripování noise).
- **Distribuce velikostí** – zjišťujeme median výšek slov, z toho odvozujeme
  průměrnou výšku “base textu”, a dělíme slova na “malá”, “typická” a “velká”.
- **Heading fonts** – podle rozdílu velikostí (`HEADING_FONT_GAP_THRESHOLD`)
  a relativního zastoupení (`HEADING_FONT_MAX_RATIO`) identifikujeme fonty,
  které slouží primárně pro nadpisy.

### Segmentace bloků a odstavců

- **Inicializace** – každý ALTO `TextBlock` se zpracuje, ale výsledný HTML
  blok může vzniknout rozdělením nebo spojením více řádků.
- **Sledování řádků** – ukládáme si pro každý řádek text, average word height,
  tokeny, bold flagy, font sizes, center, šířky a VPOS.
- **Vertikální prahy** – z mediánu mezer mezi řádky a mediánu výšek řádků
  vypočítáme kandidáty pro threshold; obě hodnoty kombinujeme a zároveň
  limitujeme maximem (ne víc než `VERTICAL_MAX_FACTOR * median_height`).
- **Horizontální prahy** – bereme medián šířek řádků a kladných posunů (po
  ořezu outlierů), z nich vytváříme kandidáty; negativní posun násobíme
  `NEGATIVE_SHIFT_MULTIPLIER`, aby byl citlivější.
- **Effective left** – první relevantní token (ignorujíc noise) určí skutečný
  HPOS řádku. Díky tomu se negativní split nevyvolá jen kvůli počáteční tečce
  nebo závorkám.
- **Horizontální split** – pokud je posun doprava větší než práh, blok se
  ukončí a otevře se nový. Tím vznikají nové odstavce u odrážek, dialogů nebo
  výčtů.
- **Negativní split (odsazení)** – když se další řádek vrátí hodně doleva,
  rozdělíme předchozí řádky na samostatné bloky a označíme je `split_reason =
  'horizontal_indent'`, aby se s nimi později zacházelo opatrněji. 
- **Centrované skupiny** – už při průchodu řádků tvoříme podskupiny se
  samostatným centered flagem. Při splitu se centered status přenáší na nový
  blok jen tehdy, pokud to dává smysl (subgroup má centered True).
- **Font/height změny** – u centrovaných bloků sledujeme poměr velikostí písma
  (`FONT_SIZE_SPLIT_RATIO`) a průměrné výšky (`CENTER_LINE_HEIGHT_RATIO`). Změna
  vyvolá split mezi částmi (např. z názvu na autora).

### Detekce nadpisů

- **Výchozí tag** – blok začíná jako `<p>`.
- **Analýza slov** – pro nominované bloky (např. s heading fontem nebo boldem)
  vytvoříme trojice (výška, délka, token). Nejprve vyhodíme čistě interpunkční
  tokeny.
- **Length filtr** – iterativně zvyšujeme minimální délku slova (od `1` dolů),
  dokud nemáme dost slov (`> WORD_LENGTH_FILTER_MIN_WORDS`). Tím ignorujeme
  jednoznakové OCR chyby.
- **Porovnání s průměrem** – spočítáme podíl slov nad `HEADING_H1_RATIO *
  average_height` a `HEADING_H2_RATIO * average_height`. Pokud je poměr nad
  minimem, povýšíme blok na `h1` nebo `h2`.
- **Ochrany** – útržky vzniklé negativním splitem (např. návrat v dialogu)
  vyžadují minimálně tři slova a blok se nepromuje, pokud začíná uvozovkou.
- **Heading font multiplier** – pokud byl blok nalezen v nadpisovém fontu,
  snižujeme potřebný poměr (`HEADING_FONT_RATIO_MULTIPLIER`).

### Detekce malého textu a poznámek

- **Stejné filtry jako headingy** – používáme shodné čištění tokenů i length
  filtry, aby se zohlednily OCR artefakty.
- **Práh** – `SMALL_MIN_WORD_RATIO` určuje, kolik slov musí být pod hranicí
  “malého” textu. Pokud byl původní tag `h3` nebo řádek začíná hvězdičkou,
  práh ulevujeme (`SMALL_RATIO_MULTIPLIER`).
- **Výstup** – bloky se označí jako `small`, aby se v HTML generovaly se
  stylovými třídami pro poznámky.

### Zacházení s čísly stránek

- **Shromažďování kandidátů** – u krátkých řádků (šířka < 20 % stránky) se
  hledá match k metadata číslu stránky (regex). Nejprve hledáme přesnou shodu,
  fallback je “suspect”.
- **Sekundární shody** – řádky typu `123*`, hvězdičkové poznámky nebo čistá
  čísla ukládáme jako sekundární kandidáty i s příznakem, zda obsahují jen
  čísla.
- **Povýšení** – když neexistuje jiný kandidát a máme jediný čistě číselný
  sekundární řádek, povýšíme ho na primární a přeskočíme ho při renderování
  (aby se číslo nezobrazovalo dvakrát).
- **Poznámky v HTML** – do výstupu přidáváme `<note>` s informací o nalezeném
  čísle stránky a OCR textu. Výrazné rozdíly signalizujeme hláškou
  `"!FOUND ONLY KANDIDATE!"`.

### Generování výstupu

- **HTML builder** – na základě výsledných bloků skládáme `<p>`, `<h1>`, `<h2>`,
  `<h3>`, `<small>`, případně `<div class="centered">` podle flags.
- **Hyphenation** – `HypPart1` a `HypPart2` se spojí do jednoho slova; druhá
  část se v textu už neobjeví, ale poslouží pro výpočet HPOS.
- **Spojování bloků** – tag `h3` po jedné tučné linii se vytvoří jako
  samostatný blok, zbytek řetězce se přesune do dalšího odstavce.
- **Výsledné funkce** – `get_formatted_text` vrací HTML, `get_full_text`
  prostý text, `get_blocks_for_reading` připraví bloky s bounding boxy pro
  zvýraznění.

### Debug režim a pomocné skripty

- `test_final.py` – stáhne konkrétní UUID, zavolá klíčové metody a uloží
  výsledky do `test_output/`.
- `test_input_api.py` – dovolí testovat různé API základny a vytiskne statistiky
  načteného kontextu.
- `batch_comparison_web.py` – iteruje přes seznam problémových UUID a spouští
  webové rozhraní pro manuální kontrolu.
- `problematic_uuid.txt` – udržuje testovací scénáře s poznámkami, které části
  pipeline máme při ladění sledovat.

---

Nová Python verze tedy pokrývá celý řetězec – od získání dat, přes robustní
statistiku a segmentaci, až po detailní heuristiky pro nadpisy, malé texty a
čísla stránek. Oproti původní TypeScript implementaci funguje na reálných
OCR datech výrazně spolehlivěji a dává nám do ruky laditelné parametry i
diagnostické logy.***
