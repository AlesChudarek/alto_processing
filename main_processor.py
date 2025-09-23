#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hlavní ALTO procesor - převedený z ORIGINAL_alto-service.ts
Podporuje české znaky a formátovaný výstup
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import html
import re
import sys
import argparse

class AltoProcessor:
    def __init__(self, iiif_base_url: str = "https://kramerius5.nkp.cz/search/iiif"):
        self.iiif_base_url = iiif_base_url

    def get_alto_data(self, uuid: str) -> str:
        """Stáhne ALTO XML data pro daný UUID"""
        url = f"https://kramerius5.nkp.cz/search/api/v5.0/item/uuid:{uuid}/streams/ALTO"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            # Explicitně dekódujeme jako UTF-8
            content = response.content.decode('utf-8')
            return content
        except Exception as e:
            print(f"Chyba při stahování ALTO dat: {e}")
            return ""

    def get_boxes(self, alto: str, query: str, width: int, height: int) -> List[List[List[int]]]:
        """Najde bounding boxy pro daný query"""
        if '~' in query:
            query = query.split('~')[0]

        boxes = []
        word_array = query.replace('"', '').split()

        try:
            root = ET.fromstring(alto)
        except ET.ParseError:
            # Zkusíme s lxml
            try:
                from lxml import etree
                root = etree.fromstring(alto.encode('utf-8'))
            except ImportError:
                print("Pro lepší parsing nainstalujte: pip install lxml")
                return []

        # Získání rozměrů
        page = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}Page')
        if page is None:
            page = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}Page')

        print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}PrintSpace')
        if print_space is None:
            print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}PrintSpace')

        alto_height = int(page.get('HEIGHT', '0'))
        alto_width = int(page.get('WIDTH', '0'))
        alto_height2 = int(print_space.get('HEIGHT', '0'))
        alto_width2 = int(print_space.get('WIDTH', '0'))

        wc = 1
        hc = 1
        # Přepočítací koeficienty pro převod ALTO souřadnic do požadované bitmapy
        if alto_height > 0 and alto_width > 0:
            wc = width / alto_width
            hc = height / alto_height
        elif alto_height2 > 0 and alto_width2 > 0:
            wc = width / alto_width2
            hc = height / alto_height2

        # Najdeme všechny String elementy
        strings = []
        for elem in root.iter():
            if elem.tag.endswith('String'):
                strings.append(elem)

        for word in word_array:
            # Normalizace hledaného slova i OCR textu (bez diakritiky se zde nepracuje)
            word = word.lower().replace('-', '').replace('?', '').replace('!', '').replace('»', '').replace('«', '').replace(';', '').replace(')', '').replace('(', '').replace('.', '').replace('„', '').replace('"', '').replace('"', '').replace(',', '').replace(')', '')

            for string_el in strings:
                content = string_el.get('CONTENT', '').lower().replace('-', '').replace('?', '').replace('!', '').replace('»', '').replace('«', '').replace(';', '').replace(')', '').replace('(', '').replace('.', '').replace('„', '').replace('"', '').replace('"', '').replace(',', '').replace(')', '')
                subs_content = string_el.get('SUBS_CONTENT', '').lower().replace('-', '').replace('?', '').replace('!', '').replace('»', '').replace('«', '').replace(';', '').replace(')', '').replace('(', '').replace('.', '').replace('„', '').replace('"', '').replace('"', '').replace(',', '').replace(')', '')

                if content == word or subs_content == word:
                    w = int(string_el.get('WIDTH', '0')) * wc
                    h = int(string_el.get('HEIGHT', '0')) * hc
                    vpos = int(string_el.get('VPOS', '0')) * hc
                    hpos = int(string_el.get('HPOS', '0')) * wc

                    box = [
                        [hpos, -vpos],
                        [hpos + w, -vpos],
                        [hpos + w, -vpos - h],
                        [hpos, -vpos - h],
                        [hpos, -vpos]
                    ]
                    boxes.append(box)

        return boxes

    def get_text_in_box(self, alto: str, box: List[int], width: int, height: int) -> str:
        """Získá text v daném bounding boxu"""
        try:
            root = ET.fromstring(alto)
        except ET.ParseError:
            try:
                from lxml import etree
                root = etree.fromstring(alto.encode('utf-8'))
            except ImportError:
                return ""

        page = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}Page')
        if page is None:
            page = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}Page')

        print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}PrintSpace')
        if print_space is None:
            print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}PrintSpace')

        alto_height = int(page.get('HEIGHT', '0'))
        alto_width = int(page.get('WIDTH', '0'))
        alto_height2 = int(print_space.get('HEIGHT', '0'))
        alto_width2 = int(print_space.get('WIDTH', '0'))

        wc = 1
        hc = 1
        if alto_height > 0 and alto_width > 0:
            wc = width / alto_width
            hc = height / alto_height
        elif alto_height2 > 0 and alto_width2 > 0:
            wc = width / alto_width2
            hc = height / alto_height2

        # Převod vstupního boxu ze zobrazovacího prostoru zpět do ALTO jednotek
        w1 = box[0] / wc
        w2 = box[2] / wc
        h1 = -box[3] / hc
        h2 = -box[1] / hc

        text = ""
        text_lines = []
        for elem in root.iter():
            if elem.tag.endswith('TextLine'):
                text_lines.append(elem)

        for text_line in text_lines:
            hpos = int(text_line.get('HPOS', '0'))
            vpos = int(text_line.get('VPOS', '0'))
            text_line_width = int(text_line.get('WIDTH', '0'))
            text_line_height = int(text_line.get('HEIGHT', '0'))

            if hpos >= w1 and hpos + text_line_width <= w2 and vpos >= h1 and vpos + text_line_height <= h2:
                strings = []
                for elem in text_line.iter():
                    if elem.tag.endswith('String'):
                        strings.append(elem)

                for string_el in strings:
                    string_hpos = int(string_el.get('HPOS', '0'))
                    string_vpos = int(string_el.get('VPOS', '0'))
                    string_width = int(string_el.get('WIDTH', '0'))
                    string_height = int(string_el.get('HEIGHT', '0'))

                    if string_hpos >= w1 and string_hpos + string_width <= w2 and string_vpos >= h1 and string_vpos + string_height <= h2:
                        # Práce s rozdělenými slovy: HypPart1/2 označuje dělení do dvou řádků
                        content = string_el.get('CONTENT', '')
                        subs_content = string_el.get('SUBS_CONTENT', '')
                        subs_type = string_el.get('SUBS_TYPE', '')

                        if subs_type == 'HypPart1':
                            content = subs_content
                        elif subs_type == 'HypPart2':
                            continue

                        text += content + ' '

        return text.strip()

    def get_full_text(self, alto: str) -> str:
        """Získá kompletní text z ALTO"""
        try:
            root = ET.fromstring(alto)
        except ET.ParseError:
            try:
                from lxml import etree
                root = etree.fromstring(alto.encode('utf-8'))
            except ImportError:
                return ""

        text = ""
        text_lines = []
        for elem in root.iter():
            if elem.tag.endswith('TextLine'):
                text_lines.append(elem)

        for text_line in text_lines:
            strings = []
            for elem in text_line.iter():
                if elem.tag.endswith('String'):
                    strings.append(elem)

            for string_el in strings:
                # Slepujeme text tak, jak jde po String elementech v rámci řádku
                content = string_el.get('CONTENT', '')
                subs_content = string_el.get('SUBS_CONTENT', '')
                subs_type = string_el.get('SUBS_TYPE', '')

                if subs_type == 'HypPart1':
                    content = subs_content
                elif subs_type == 'HypPart2':
                    continue

                text += content + ' '

        return text.strip()

    def get_blocks_for_reading(self, alto: str) -> List[Dict]:
        """Získá bloky textu pro čtení"""
        try:
            root = ET.fromstring(alto)
        except ET.ParseError:
            try:
                from lxml import etree
                root = etree.fromstring(alto.encode('utf-8'))
            except ImportError:
                return []

        page = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}Page')
        if page is None:
            page = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}Page')

        print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}PrintSpace')
        if print_space is None:
            print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}PrintSpace')

        if print_space is None:
            return []

        alto_height = int(page.get('HEIGHT', '0'))
        alto_width = int(page.get('WIDTH', '0'))
        alto_height2 = int(print_space.get('HEIGHT', '0'))
        alto_width2 = int(print_space.get('WIDTH', '0'))

        aw = alto_width if alto_height > 0 and alto_width > 0 else alto_width2
        ah = alto_height if alto_height > 0 and alto_width > 0 else alto_height2

        blocks = []
        # Během průchodu skládáme aktuální blok; bbox slouží pro případné zvýraznění
        block = {'text': '', 'hMin': 0, 'hMax': 0, 'vMin': 0, 'vMax': 0, 'width': aw, 'height': ah}

        text_lines = []
        for elem in root.iter():
            if elem.tag.endswith('TextLine'):
                text_lines.append(elem)

        lines = 0
        last_bottom = 0

        for text_line in text_lines:
            text_line_width = int(text_line.get('WIDTH', '0'))
            if text_line_width < 50:
                continue

            text_line_height = int(text_line.get('HEIGHT', '0'))
            text_line_vpos = int(text_line.get('VPOS', '0'))
            bottom = text_line_vpos + text_line_height
            diff = text_line_vpos - last_bottom

            if last_bottom > 0 and diff > 50:
                # Větší mezera značí nový odstavec; mezi bloky vkládáme oddělovač
                if block['text']:
                    block['text'] += '. -- -- '

            last_bottom = bottom
            lines += 1

            strings = []
            for elem in text_line.iter():
                if elem.tag.endswith('String'):
                    strings.append(elem)

            for string_el in strings:
                string_hpos = int(string_el.get('HPOS', '0'))
                string_vpos = int(string_el.get('VPOS', '0'))
                string_width = int(string_el.get('WIDTH', '0'))
                string_height = int(string_el.get('HEIGHT', '0'))

                if block['hMin'] == 0 or block['hMin'] > string_hpos:
                    block['hMin'] = string_hpos
                if block['hMax'] == 0 or block['hMax'] < string_hpos + string_width:
                    block['hMax'] = string_hpos + string_width
                if block['vMin'] == 0 or block['vMin'] > string_vpos:
                    block['vMin'] = string_vpos
                if block['vMax'] == 0 or block['vMax'] < string_vpos + string_height:
                    block['vMax'] = string_vpos + string_height

                content = string_el.get('CONTENT', '')
                block['text'] += content

                if lines >= 3 and len(block['text']) > 120 and (content.endswith('.') or content.endswith(';')):
                    # Delší souvislý text uzavíráme do samostatného bloku
                    if block['text']:
                        blocks.append(block)
                    block = {'text': '', 'hMin': 0, 'hMax': 0, 'vMin': 0, 'vMax': 0, 'width': aw, 'height': ah}
                    lines = 0
                else:
                    block['text'] += ' '

        if block['text']:
            blocks.append(block)

        return blocks

    def get_formatted_text(self, alto: str, uuid: str, width: int, height: int) -> str:
        """Hlavní funkce pro formátovaný text - převedená z TypeScript"""
        try:
            from lxml import etree
            root = etree.fromstring(alto.encode('utf-8'))
        except ImportError:
            try:
                root = ET.fromstring(alto)
            except ET.ParseError:
                return ""

        # Získání rozměrů
        page = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}Page')
        if page is None:
            page = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}Page')

        print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v3#}PrintSpace')
        if print_space is None:
            print_space = root.find('.//{http://www.loc.gov/standards/alto/ns-v2#}PrintSpace')

        if print_space is None:
            return ""

        alto_height = int(page.get('HEIGHT', '0'))
        alto_width = int(page.get('WIDTH', '0'))
        alto_height2 = int(print_space.get('HEIGHT', '0'))
        alto_width2 = int(print_space.get('WIDTH', '0'))

        wc = 1
        hc = 1
        if alto_height > 0 and alto_width > 0 and width > 0 and height > 0:
            wc = width / alto_width
            hc = height / alto_height
        elif alto_height2 > 0 and alto_width2 > 0 and width > 0 and height > 0:
            wc = width / alto_width2
            hc = height / alto_height2

        # Parsování fontů: mapujeme ID stylů na jejich velikost písma
        fonts = {}
        styles = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextStyle')
        if not styles:
            styles = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextStyle')

        for style in styles:
            style_id = style.get('ID')
            font_size = style.get('FONTSIZE')
            if style_id and font_size:
                fonts[style_id] = int(font_size)

        blocks = []

        # Zpracování TextBlocků
        text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextBlock')
        if not text_blocks:
            text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextBlock')

        for block_elem in text_blocks:
            style_refs = block_elem.get('STYLEREFS', '')
            tag = 'p'

            if ' ' in style_refs:
                parts = style_refs.split()
                if len(parts) > 1:
                    font_id = parts[1]
                    size = fonts.get(font_id, 0)
                    # Hrubá heuristika pro převod větších fontů na nadpisy
                    if size > 18:
                        tag = 'h1'
                    elif size > 11:
                        tag = 'h2'

            text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextLine')
            if not text_lines:
                text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextLine')

            current_block = {'text': '', 'tag': tag}
            lines = 0
            last_bottom = 0
            last_left = 0
            all_bold = True

            for text_line in text_lines:
                text_line_width = int(text_line.get('WIDTH', '0'))
                if text_line_width < 50:
                    continue

                text_line_height = int(text_line.get('HEIGHT', '0'))
                text_line_vpos = int(text_line.get('VPOS', '0'))
                text_line_hpos = int(text_line.get('HPOS', '0'))
                bottom = text_line_vpos + text_line_height
                v_diff = text_line_vpos - last_bottom

                if last_bottom > 0 and v_diff > 40:
                    # Velká vertikální mezera = nový blok textu
                    if current_block['text']:
                        blocks.append(current_block)
                    current_block = {'text': '', 'tag': tag}
                    lines = 0

                last_bottom = bottom

                h_diff = text_line_hpos - last_left
                if last_left > 0 and h_diff > 40:
                    # Podobně reagujeme na výrazný horizontální posun (např. tabulky, sloupce)
                    if current_block['text']:
                        blocks.append(current_block)
                    current_block = {'text': '', 'tag': tag}
                    lines = 0

                last_left = text_line_hpos

                strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not strings:
                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')

                for string_el in strings:
                    style = string_el.get('STYLE', '')
                    if not style or 'bold' not in style:
                        all_bold = False

                    content = string_el.get('CONTENT', '')
                    subs_content = string_el.get('SUBS_CONTENT', '')
                    subs_type = string_el.get('SUBS_TYPE', '')

                    # Dekódování HTML entit pro správné zobrazení českých znaků
                    content = html.unescape(content)
                    subs_content = html.unescape(subs_content)

                    if subs_type == 'HypPart1':
                        content = subs_content
                    elif subs_type == 'HypPart2':
                        continue

                    current_block['text'] += content + ' '

                lines += 1

                # Logika pro <h3> - shodná s původní TypeScript implementací
                if lines == 1 and all_bold and current_block['text']:
                    current_block['tag'] = 'h3'
                    blocks.append(current_block)
                    current_block = {'text': '', 'tag': tag}
                    lines = 0
                else:
                    all_bold = True

            if current_block['text']:
                blocks.append(current_block)

        # Generování HTML - přesně jako v TypeScript
        result = ""
        for block in blocks:
            if block['text'].strip():  # Jen neprázdné bloky
                result += f"<{block['tag']}>{block['text'].strip()}</{block['tag']}>"

        return result

def main():
    parser = argparse.ArgumentParser(description='ALTO Processor - zpracování ALTO XML z Kramerius')
    parser.add_argument('uuid', help='UUID dokumentu')
    parser.add_argument('--width', type=int, default=800, help='Šířka výstupu (výchozí: 800)')
    parser.add_argument('--height', type=int, default=1200, help='Výška výstupu (výchozí: 1200)')
    parser.add_argument('--output', default='output.txt', help='Výstupní soubor (výchozí: output.txt)')

    args = parser.parse_args()

    processor = AltoProcessor()

    print(f"Zpracovávám UUID: {args.uuid}")

    # Stáhnout ALTO data
    alto_xml = processor.get_alto_data(args.uuid)
    if not alto_xml:
        print("Nepodařilo se stáhnout ALTO data")
        return

    print(f"Staženo {len(alto_xml)} znaků ALTO XML")

    # Zpracovat a získat formátovaný text
    formatted_text = processor.get_formatted_text(alto_xml, args.uuid, args.width, args.height)

    if not formatted_text:
        print("Nepodařilo se zpracovat text")
        return

    print(f"Zpracováno {len(formatted_text)} znaků textu")

    # Uložit výsledek
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(formatted_text)

    print(f"Výsledek uložen do: {args.output}")

    # Zobrazit náhled
    print("\n=== NÁHLED VÝSLEDKU ===")
    print(formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text)

if __name__ == "__main__":
    main()
