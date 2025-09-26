#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hlavní ALTO procesor - převedený z ORIGINAL_alto-service.ts
Podporuje české znaky a formátovaný výstup
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Any
import html
import re
import sys
import argparse
import statistics

# Heuristické multiplikátory pro dělení bloků; úprava na jednom místě usnadní ladění.
VERTICAL_GAP_MULTIPLIER = 2.5   # Kolikrát musí být mezera mezi řádky větší než typická mezera, aby vznikl nový blok.
VERTICAL_HEIGHT_RATIO = 0.85    # Poměr k mediánu výšky řádku, přispívá k prahu pro rozdělení bloku.
VERTICAL_MAX_FACTOR = 3         # Horní limit pro vertikální práh v násobcích mediánu výšky řádku.
HORIZONTAL_WIDTH_RATIO = 0.012  # Kandidát prahu = medián šířek řádků * tato hodnota (nižší = citlivější).
HORIZONTAL_SHIFT_MULTIPLIER = 0.85  # Kandidát prahu = medián kladných posunů * tato hodnota.
HORIZONTAL_MIN_THRESHOLD = 12   # Minimální povolený práh pro horizontální dělení (v ALTO jednotkách).
NEGATIVE_SHIFT_MULTIPLIER = 0.85  # Negativní hranice = horizontální práh * tato hodnota (víc citlivá než pozitivní).
FALLBACK_THRESHOLD = 40        # Záložní hodnota, pokud nelze heuristiku spočítat.

class AltoProcessor:
    def __init__(
        self,
        iiif_base_url: str = "https://kramerius5.nkp.cz/search/iiif",
        api_base_url: str = "https://kramerius5.nkp.cz/search/api/v5.0",
    ):
        self.iiif_base_url = iiif_base_url
        self.api_base_url = api_base_url

    @staticmethod
    def _strip_uuid_prefix(value: Optional[str]) -> str:
        if not value:
            return ""
        return value.split(":", 1)[1] if value.startswith("uuid:") else value

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        if not value:
            return ""
        cleaned = value.replace("\xa0", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    def get_item_json(self, uuid: str) -> Dict[str, Any]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return {}

        url = f"{self.api_base_url}/item/uuid:{normalized}"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"Chyba při načítání metadat objektu {uuid}: {exc}")
            return {}

    def get_children(self, uuid: str) -> List[Dict[str, Any]]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return []

        url = f"{self.api_base_url}/item/uuid:{normalized}/children"
        try:
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            print(f"Chyba při načítání potomků objektu {uuid}: {exc}")
            return []

    def get_mods_metadata(self, uuid: str) -> List[Dict[str, str]]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return []

        url = f"{self.api_base_url}/item/uuid:{normalized}/streams/BIBLIO_MODS"
        try:
            response = requests.get(url, timeout=25)
            response.raise_for_status()
        except Exception as exc:
            print(f"Chyba při načítání MODS metadat {uuid}: {exc}")
            return []

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            print(f"Chyba při parsování MODS metadat {uuid}: {exc}")
            return []

        ns = {"mods": "http://www.loc.gov/mods/v3"}
        record = root.find('.//mods:mods', ns)
        if record is None and root.tag.endswith('mods'):
            record = root
        if record is None:
            return []

        metadata: List[Dict[str, str]] = []

        def add_entry(label: str, value: str) -> None:
            cleaned = self._clean_text(value)
            if cleaned:
                metadata.append({"label": label, "value": cleaned})

        titles = []
        for title_info in record.findall('mods:titleInfo', ns):
            main_title = (title_info.findtext('mods:title', default='', namespaces=ns) or '').strip()
            subtitle = (title_info.findtext('mods:subTitle', default='', namespaces=ns) or '').strip()
            part_number = (title_info.findtext('mods:partNumber', default='', namespaces=ns) or '').strip()
            part_name = (title_info.findtext('mods:partName', default='', namespaces=ns) or '').strip()
            segments = [seg for seg in [main_title, subtitle] if seg]
            if part_number or part_name:
                part_segments = " ".join(filter(None, [part_number, part_name])).strip()
                if part_segments:
                    segments.append(part_segments)
            title_text = " - ".join(segments).strip()
            if title_text:
                titles.append(title_text)
        if titles:
            add_entry("Název", '; '.join(dict.fromkeys(titles)))

        authors = []
        for name_el in record.findall('mods:name', ns):
            name_parts = []
            dates = []
            for part in name_el.findall('mods:namePart', ns):
                text = (part.text or '').strip()
                if not text:
                    continue
                if part.attrib.get('type') == 'date':
                    dates.append(text)
                else:
                    name_parts.append(text)
            if not name_parts:
                continue
            author_text = ' '.join(name_parts)
            if dates:
                author_text = f"{author_text} ({', '.join(dates)})"
            role_terms = []
            for role_term in name_el.findall('.//mods:roleTerm', ns):
                term_text = (role_term.text or '').strip()
                if term_text:
                    role_terms.append(term_text)
            if role_terms:
                author_text = f"{author_text} [{', '.join(dict.fromkeys(role_terms))}]"
            authors.append(author_text)
        if authors:
            add_entry("Autoři", '; '.join(dict.fromkeys(authors)))

        for origin_info in record.findall('mods:originInfo', ns):
            publisher = (origin_info.findtext('mods:publisher', default='', namespaces=ns) or '').strip()
            place = (origin_info.findtext('mods:place/mods:placeTerm', default='', namespaces=ns) or '').strip()
            date_issued = (origin_info.findtext('mods:dateIssued', default='', namespaces=ns) or '').strip()
            edition = (origin_info.findtext('mods:edition', default='', namespaces=ns) or '').strip()
            if edition:
                add_entry("Edice", edition)
            publication_segments = [segment for segment in [place, publisher, date_issued] if segment]
            if publication_segments:
                add_entry("Publikační údaje", ', '.join(publication_segments))

        languages = []
        for language in record.findall('mods:language/mods:languageTerm', ns):
            lang_text = (language.text or '').strip()
            if lang_text:
                languages.append(lang_text)
        if languages:
            add_entry("Jazyk", ', '.join(dict.fromkeys(languages)))

        extents = []
        for extent in record.findall('mods:physicalDescription/mods:extent', ns):
            text = (extent.text or '').strip()
            if text:
                extents.append(text)
        if extents:
            add_entry("Rozsah", '; '.join(dict.fromkeys(extents)))

        identifiers = []
        for identifier in record.findall('mods:identifier', ns):
            text = (identifier.text or '').strip()
            if not text:
                continue
            id_type = identifier.attrib.get('type')
            if id_type:
                identifiers.append(f"{id_type}: {text}")
            else:
                identifiers.append(text)
        if identifiers:
            add_entry("Identifikátory", '; '.join(dict.fromkeys(identifiers)))

        notes = []
        for note in record.findall('mods:note', ns):
            text = (note.text or '').strip()
            if text:
                notes.append(text)
        if notes:
            add_entry("Poznámky", ' | '.join(dict.fromkeys(notes)))

        subjects = []
        for subject in record.findall('mods:subject', ns):
            terms = []
            for child in subject:
                text = (child.text or '').strip()
                if text:
                    terms.append(text)
            if terms:
                subjects.append(' - '.join(terms))
        if subjects:
            add_entry("Témata", '; '.join(dict.fromkeys(subjects)))

        return metadata

    def _page_summary_from_child(self, child: Dict[str, Any], index: int) -> Dict[str, Any]:
        details = child.get('details') or {}
        return {
            "uuid": self._strip_uuid_prefix(child.get('pid')),
            "index": index,
            "title": self._clean_text(child.get('title')),
            "pageNumber": self._clean_text(details.get('pagenumber')),
            "pageType": self._clean_text(details.get('type')),
            "pageSide": self._clean_text(details.get('pageposition') or details.get('pagePosition') or details.get('pagerole')),
            "model": child.get('model'),
            "policy": child.get('policy'),
        }

    def collect_book_pages(self, book_uuid: str, max_depth: int = 4) -> List[Dict[str, Any]]:
        visited: set[str] = set()
        pages: List[Dict[str, Any]] = []

        def walk(node_uuid: str, depth: int) -> None:
            if depth > max_depth or not node_uuid or node_uuid in visited:
                return
            visited.add(node_uuid)

            children = self.get_children(node_uuid)
            for child in children:
                child_uuid = self._strip_uuid_prefix(child.get('pid'))
                if not child_uuid:
                    continue
                model = child.get('model')
                if model == 'page':
                    summary = self._page_summary_from_child(child, len(pages))
                    pages.append(summary)
                elif depth + 1 <= max_depth:
                    walk(child_uuid, depth + 1)

        walk(self._strip_uuid_prefix(book_uuid), 0)
        return pages

    def get_book_context(self, item_uuid: str) -> Optional[Dict[str, Any]]:
        """Vrátí metadata knihy, seznam stran a aktuální stranu pro zadaný UUID."""

        item_data = self.get_item_json(item_uuid)
        if not item_data:
            return None

        model = item_data.get('model')
        page_data: Optional[Dict[str, Any]] = None

        if model == 'page':
            page_data = item_data
            root_pid = item_data.get('root_pid') or ''
            book_uuid = self._strip_uuid_prefix(root_pid)
            if not book_uuid and item_data.get('context'):
                context_path = item_data['context'][0]
                if context_path:
                    book_uuid = self._strip_uuid_prefix(context_path[0].get('pid'))
        else:
            book_uuid = self._strip_uuid_prefix(item_data.get('pid'))

        if not book_uuid:
            return None

        book_data = item_data if model != 'page' else self.get_item_json(book_uuid)
        if not book_data:
            return None

        pages = self.collect_book_pages(book_uuid)
        if not pages:
            print(f"Pro knihu {book_uuid} se nepodařilo načíst žádné strany")

        page_uuid = self._strip_uuid_prefix((page_data or {}).get('pid')) or self._strip_uuid_prefix(item_uuid)
        if page_data is None and pages:
            page_uuid = pages[0]['uuid']
            page_data = self.get_item_json(page_uuid)

        current_index = -1
        if pages:
            for entry in pages:
                if entry.get('uuid') == page_uuid:
                    current_index = entry.get('index', pages.index(entry))
                    break
        page_summary = None
        if current_index >= 0:
            page_summary = pages[current_index]
        elif page_data:
            page_summary = {
                "uuid": page_uuid,
                "index": current_index if current_index >= 0 else 0,
                "title": self._clean_text(page_data.get('title')),
                "pageNumber": self._clean_text((page_data.get('details') or {}).get('pagenumber')),
                "pageType": self._clean_text((page_data.get('details') or {}).get('type')),
                "pageSide": self._clean_text((page_data.get('details') or {}).get('pageposition') or (page_data.get('details') or {}).get('pagePosition') or (page_data.get('details') or {}).get('pagerole')),
                "model": page_data.get('model'),
                "policy": page_data.get('policy'),
            }

        resolved_index = current_index if current_index >= 0 else (page_summary.get('index') if page_summary else -1)

        return {
            "book_uuid": book_uuid,
            "book": book_data,
            "page_uuid": page_uuid,
            "page": page_summary,
            "page_item": page_data,
            "pages": pages,
            "mods": self.get_mods_metadata(book_uuid),
            "current_index": resolved_index,
        }

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

        def finalize_block(block_data):
            """Po spojení řádků uloží blok, pokud není prázdný."""
            text_content = ' '.join(block_data['lines']).strip()
            if text_content:
                blocks.append({'text': text_content, 'tag': block_data['tag']})

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

            line_records = []
            for text_line in text_lines:
                text_line_width = int(text_line.get('WIDTH', '0'))
                if text_line_width < 50:
                    continue

                text_line_height = int(text_line.get('HEIGHT', '0'))
                text_line_vpos = int(text_line.get('VPOS', '0'))
                text_line_hpos = int(text_line.get('HPOS', '0'))

                line_records.append({
                    'element': text_line,
                    'width': text_line_width,
                    'height': text_line_height,
                    'vpos': text_line_vpos,
                    'hpos': text_line_hpos,
                    'bottom': text_line_vpos + text_line_height
                })

            if not line_records:
                continue

            line_heights = [record['height'] for record in line_records]
            line_widths = [record['width'] for record in line_records]

            vertical_gaps = []
            horizontal_shifts = []
            previous_bottom = None
            previous_left = None

            for record in line_records:
                if previous_bottom is not None:
                    gap = record['vpos'] - previous_bottom
                    if gap > 0:
                        vertical_gaps.append(gap)

                if previous_left is not None:
                    shift = record['hpos'] - previous_left
                    if shift > 0:
                        horizontal_shifts.append(shift)

                previous_bottom = record['bottom']
                previous_left = record['hpos']

            median_height = statistics.median(line_heights) if line_heights else 0
            positive_gaps = [gap for gap in vertical_gaps if gap > 0]
            median_gap = statistics.median(positive_gaps) if positive_gaps else 0

            vertical_threshold_candidates = []
            if median_gap:
                vertical_threshold_candidates.append(median_gap * VERTICAL_GAP_MULTIPLIER)
            if median_height:
                vertical_threshold_candidates.append(median_height * VERTICAL_HEIGHT_RATIO)

            if vertical_threshold_candidates:
                vertical_threshold = max(int(round(value)) for value in vertical_threshold_candidates)
                vertical_threshold = max(vertical_threshold, 1)
                if median_height:
                    vertical_threshold = min(vertical_threshold, int(round(median_height * VERTICAL_MAX_FACTOR)))
            else:
                vertical_threshold = FALLBACK_THRESHOLD

            median_width = statistics.median(line_widths) if line_widths else 0
            trimmed_shifts = []
            if horizontal_shifts:
                sorted_shifts = sorted(horizontal_shifts)
                cutoff = max(1, int(len(sorted_shifts) * 0.5))
                trimmed_shifts = sorted_shifts[:cutoff]

            median_shift = statistics.median(trimmed_shifts) if trimmed_shifts else 0
            if median_width and median_shift and median_shift > median_width * 0.6:
                median_shift = 0

            horizontal_threshold_candidates = []
            if median_width:
                horizontal_threshold_candidates.append(median_width * HORIZONTAL_WIDTH_RATIO)
            if median_shift:
                horizontal_threshold_candidates.append(median_shift * HORIZONTAL_SHIFT_MULTIPLIER)

            if horizontal_threshold_candidates:
                horizontal_threshold = max(int(round(value)) for value in horizontal_threshold_candidates)
                horizontal_threshold = max(horizontal_threshold, HORIZONTAL_MIN_THRESHOLD)
            else:
                horizontal_threshold = FALLBACK_THRESHOLD

            current_block = {'lines': [], 'tag': tag}
            lines = 0
            last_bottom = None
            last_left = None

            for record in line_records:
                text_line = record['element']
                text_line_vpos = record['vpos']
                text_line_hpos = record['hpos']
                bottom = record['bottom']

                if last_bottom is not None:
                    v_diff = text_line_vpos - last_bottom
                    if v_diff > vertical_threshold:
                        # Velká vertikální mezera = nový blok textu
                        if current_block['lines']:
                            finalize_block(current_block)
                        current_block = {'lines': [], 'tag': tag}
                        lines = 0

                last_bottom = bottom

                if last_left is not None:
                    h_diff = text_line_hpos - last_left
                    if h_diff > horizontal_threshold:
                        # Podobně reagujeme na výrazný horizontální posun (např. tabulky, sloupce)
                        if current_block['lines']:
                            finalize_block(current_block)
                        current_block = {'lines': [], 'tag': tag}
                        lines = 0
                    elif h_diff < 0 and abs(h_diff) > horizontal_threshold * NEGATIVE_SHIFT_MULTIPLIER and current_block['lines']:
                        if len(current_block['lines']) > 1:
                            for previous_line in current_block['lines'][:-1]:
                                finalize_block({'lines': [previous_line], 'tag': current_block['tag']})
                            current_block = {'lines': [current_block['lines'][-1]], 'tag': current_block['tag']}
                        lines = len(current_block['lines'])

                last_left = text_line_hpos

                strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not strings:
                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')

                line_parts = []
                line_all_bold = True

                for string_el in strings:
                    style = string_el.get('STYLE', '')
                    if not style or 'bold' not in style:
                        line_all_bold = False

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

                    if content:
                        line_parts.append(content)

                line_text = ' '.join(line_parts).strip()
                if not line_text:
                    continue

                prospective_count = lines + 1

                # Logika pro <h3> - shodná s původní TypeScript implementací
                if prospective_count == 1 and line_all_bold:
                    finalize_block({'lines': [line_text], 'tag': 'h3'})
                    current_block = {'lines': [], 'tag': tag}
                    lines = 0
                    last_left = text_line_hpos
                    last_bottom = bottom
                    continue

                current_block['lines'].append(line_text)
                lines = len(current_block['lines'])

            if current_block['lines']:
                finalize_block(current_block)

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
