# ALTO Processing - Python Implementation

Toto prostředí slouží pro zpracování ALTO XML souborů z Kramerius systému a jejich převod na formátovaný text s podporou českých znaků.

## Funkce

- **ALTO XML parsing**: Zpracování ALTO XML souborů s OCR daty
- **Text formatting**: Převod na strukturovaný text s nadpisy a odstavci
- **Czech character support**: Správné zobrazení českých znaků (ř, š, č, ž, ý, á, í, é, ů)
- **Web comparison**: Webové rozhraní pro porovnání s původním TypeScript kódem
- **Command line interface**: Jednoduché spuštění s UUID parametrem

## Struktura projektu

```
alto_processing/
├── main_processor.py      # Hlavní procesor ALTO XML (převedený z TypeScript)
├── comparison_web.py      # Webový server pro porovnání výsledků
├── ORIGINAL_alto-service.ts # Původní TypeScript implementace pro referenci
├── requirements.txt       # Python závislosti
└── README.md             # Tento soubor
```

## Instalace

1. **Vytvořte virtuální prostředí:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Na Windows: venv\Scripts\activate
   ```

2. **Nainstalujte závislosti:**
   ```bash
   pip install -r requirements.txt
   ```

## Použití

### 1. Příkazový řádek

```bash
# Základní zpracování s UUID
python main_processor.py "673320dd-0071-4a03-bf82-243ee206bc0b"

# S vlastními rozměry
python main_processor.py "673320dd-0071-4a03-bf82-243ee206bc0b" --width 1200 --height 900

# Uložení do specifického souboru
python main_processor.py "673320dd-0071-4a03-bf82-243ee206bc0b" --output vysledek.txt
```

### 2. Webové rozhraní pro porovnání

```bash
# Spusťte webový server
python comparison_web.py

# Otevřete v prohlížeči
# http://localhost:8000
```

Webové rozhraní umožňuje:
- Zadání UUID dokumentu
- Nastavení rozměrů
- Porovnání výstupů Python a TypeScript implementace
- Zobrazení výsledků vedle sebe

## Parametry

- **uuid**: Jedinečný identifikátor dokumentu v Kramerius systému (povinný)
- **width**: Požadovaná šířka výstupu v pixelech (výchozí: 800)
- **height**: Požadovaná výška výstupu v pixelech (výchozí: 1200)
- **output**: Výstupní soubor pro uložení výsledku (výchozí: output.txt)

## Příklady

### Zpracování Babičky Boženy Němcové

```bash
python main_processor.py "673320dd-0071-4a03-bf82-243ee206bc0b"
```

### Webové porovnání

```bash
python comparison_web.py
# Pak otevřete http://localhost:8000 a zadejte UUID
```

## Technické detaily

- **Vstup**: ALTO XML verze 2.0+ z Kramerius systému
- **Výstup**: HTML text s českým kódováním UTF-8
- **Podporované znaky**: Kompletní UTF-8 včetně české diakritiky
- **Skalování**: Proporcionální změna velikosti podle zadaných rozměrů

## Implementace

Kód je převeden z původního TypeScript kódu (`ORIGINAL_alto-service.ts`) s těmito funkcemi:
- `get_formatted_text()`: Hlavní funkce pro formátování textu
- `get_full_text()`: Získání kompletního textu
- `get_boxes()`: Nalezení bounding boxů pro vyhledávání
- `get_text_in_box()`: Získání textu v bounding boxu
- `get_blocks_for_reading()`: Rozdělení textu na čtecí bloky

## Řešení problémů

### Problémy s kódováním českých znaků

- Všechny soubory používají UTF-8 kódování
- Webové rozhraní správně zobrazuje češtinu
- Při problémech zkontrolujte nastavení terminálu

### Chyba při stahování ALTO XML

- Zkontrolujte, zda je UUID platný
- Zkontrolujte připojení k internetu
- Některé dokumenty mohou mít omezený přístup

## Přispívání

1. Forkněte projekt
2. Vytvořte feature branch
3. Implementujte změny
4. Otestujte s různými UUID
5. Vytvořte pull request

## Licence

MIT License - viz LICENSE soubor pro detaily.
