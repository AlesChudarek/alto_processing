#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script pro ověření funkčnosti ALTO procesoru
"""

import os
from main_processor import AltoProcessor

def test_processor():
    """Test hlavních funkcí procesoru"""
    processor = AltoProcessor()

    # Test UUID
    test_uuid = "93373db0-270c-11e5-855a-5ef3fc9ae867"

    print("=== Test ALTO Procesoru ===")
    print(f"Testuji s UUID: {test_uuid}")

    # Stáhnout ALTO data
    alto_xml = processor.get_alto_data(test_uuid)
    if not alto_xml:
        print("❌ Nepodařilo se stáhnout ALTO data")
        return False

    print(f"✅ Staženo {len(alto_xml)} znaků ALTO XML")

    # Test formátovaného textu
    formatted_text = processor.get_formatted_text(alto_xml, test_uuid, 800, 1200)
    if not formatted_text:
        print("❌ Nepodařilo se zpracovat text")
        return False

    print(f"✅ Zpracováno {len(formatted_text)} znaků textu")

    # Test kompletního textu
    full_text = processor.get_full_text(alto_xml)
    if not full_text:
        print("❌ Nepodařilo se získat kompletní text")
        return False

    print(f"✅ Získán kompletní text ({len(full_text)} znaků)")

    # Kontrola českých znaků
    if "Ř" in formatted_text or "ř" in formatted_text:
        print("✅ České znaky se zobrazují správně")
    else:
        print("⚠️  České znaky mohou mít problém s kódováním")
        print("Prvních 200 znaků:")
        print(repr(formatted_text[:200]))

    # Vytvoření adresáře pro výstupy, pokud neexistuje
    os.makedirs('test_output', exist_ok=True)

    # Uložení výsledků
    with open(os.path.join('test_output', 'test_formatted.html'), 'w', encoding='utf-8') as f:
        f.write(formatted_text)

    with open(os.path.join('test_output', 'test_full.txt'), 'w', encoding='utf-8') as f:
        f.write(full_text)

    print("✅ Výsledky uloženy do test_output/test_formatted.html a test_output/test_full.txt")
    print("\n=== NÁHLED ===")
    print(formatted_text[:300] + "...")

    return True

if __name__ == "__main__":
    success = test_processor()
    if success:
        print("\n🎉 Všechny testy prošly úspěšně!")
        print("\nSpuštění:")
        print("python main_processor.py <UUID>  # Zpracování s UUID")
        print("python comparison_web.py         # Webové rozhraní")
    else:
        print("\n❌ Některé testy selhaly")
