#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hlavní ALTO procesor - převedený z ORIGINAL_alto-service.ts
Podporuje české znaky a formátovaný výstup
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import html
import re
import sys
import argparse
import statistics
import json

# Heuristické multiplikátory pro dělení bloků; úprava na jednom místě usnadní ladění.
VERTICAL_GAP_MULTIPLIER = 2.5   # Kolikrát musí být mezera mezi řádky větší než typická mezera, aby vznikl nový blok.
VERTICAL_HEIGHT_RATIO = 0.85    # Poměr k mediánu výšky řádku, přispívá k prahu pro rozdělení bloku.
VERTICAL_MAX_FACTOR = 3         # Horní limit pro vertikální práh v násobcích mediánu výšky řádku.
HORIZONTAL_WIDTH_RATIO = 0.045  # Kandidát prahu = medián šířek řádků * tato hodnota (nižší = citlivější).
HORIZONTAL_SHIFT_MULTIPLIER = 0.85  # Kandidát prahu = medián kladných posunů * tato hodnota.
HORIZONTAL_MIN_THRESHOLD = 12   # Minimální povolený práh pro horizontální dělení (v ALTO jednotkách).
NEGATIVE_SHIFT_MULTIPLIER = 0.85  # Negativní hranice = horizontální práh * tato hodnota (víc citlivá než pozitivní).
FALLBACK_THRESHOLD = 40        # Záložní hodnota, pokud nelze heuristiku spočítat.

CENTER_ALIGNMENT_ERROR_MARGIN = 0.05  # 5% margin for center alignment detection relative to median line width
CENTER_ALIGNMENT_MIN_LINE_LEN_DIFF = 0.1  # 10% minimum difference between shortest and longest line for center alignment

# Konstanty pro rozhodování o nadpisech na základě výšky slov
HEADING_H2_RATIO = 1.08         # Práh pro h2: 1.08 * průměrná výška
HEADING_H1_RATIO = 2         # Práh pro h1: 1.6 * průměrná výška
HEADING_MIN_WORD_RATIO_DEFAULT = 0.81   # Výchozí minimální podíl slov v bloku, které musí překročit práh (margin of error pro OCR)
HEADING_FONT_GAP_THRESHOLD = 1.2        # Práh pro rozdíl ve velikosti fontu pro identifikaci nadpisových fontů
HEADING_FONT_RATIO_MULTIPLIER = 0.5     # Koeficient pro snížení prahu pro bloky s nadpisovými fonty
HEADING_FONT_MAX_RATIO = 0.4            # Maximální podíl řádků s fontem, aby byl považován za nadpisový

# Centralized HTTP timeouts (seconds)
API_TIMEOUT = 25
CHILDREN_TIMEOUT = 25
MODS_TIMEOUT = 25
ALTO_TIMEOUT = 30

# Nastavení pro analýzu typického formátu základního textu v rámci knihy.
TEXT_SAMPLE_WAVE_SIZE = 10           # Kolik stran z každé vlny odečíst.
TEXT_SAMPLE_MAX_WAVES = 3           # Maximální počet vln načítání.
MIN_CONFIDENCE_FOR_EARLY_STOP = 75  # Pokud confidence (v %) dosáhne této hodnoty, další vlny nejsou třeba.
BLOCK_MIN_TOTAL_CHARS = 40          # Minimální množství znaků v TextBlocku, aby šel do analýzy.
MIN_WORDS_PER_PAGE = 20             # Minimální počet slov na stránce pro výpočet průměrné výšky.

BOOK_TEXT_STYLE_CACHE: Dict[str, Dict[str, Any]] = {}

class AltoProcessor:
    def __init__(
        self,
        iiif_base_url: str = "https://kramerius5.nkp.cz/search/iiif",
        api_base_url: str = "https://kramerius5.nkp.cz/search/api/v5.0",
    ):
        self.iiif_base_url = iiif_base_url
        self.api_base_url = api_base_url
        self._book_text_cache = BOOK_TEXT_STYLE_CACHE
        # HTTP session with retries for robustness on flaky Kramerius endpoints
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

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

    @staticmethod
    def _safe_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _round_float(value: Optional[float], digits: int = 2) -> Optional[float]:
        if value is None:
            return None
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compute_confidence(counter: Counter) -> Optional[int]:
        if not counter:
            return None
        total = sum(counter.values())
        if total <= 0:
            return None
        top = counter.most_common(1)[0][1]
        return int(round((top / total) * 100))

    @staticmethod
    def _parse_alto_root(alto: str):
        if not alto:
            return None
        try:
            from lxml import etree  # type: ignore
            return etree.fromstring(alto.encode('utf-8'))
        except ImportError:
            pass
        except Exception:
            return None
        try:
            return ET.fromstring(alto)
        except ET.ParseError:
            return None
        except Exception:
            return None

    def _extract_text_styles(self, root) -> Dict[str, Dict[str, Any]]:
        styles: Dict[str, Dict[str, Any]] = {}
        if root is None:
            return styles
        for elem in root.iter():
            tag = getattr(elem, 'tag', '')
            if isinstance(tag, str) and tag.endswith('TextStyle'):
                style_id = elem.get('ID')
                if not style_id:
                    continue
                styles[style_id] = {
                    'font_size': self._safe_float(elem.get('FONTSIZE')),
                    'font_family': self._clean_text(elem.get('FONTFAMILY')),
                    'font_style': self._clean_text(elem.get('FONTSTYLE')),
                    'font_weight': self._clean_text(elem.get('FONTWEIGHT')),
                }
        return styles

    def _string_style_signature(self, string_el, fonts: Dict[str, Dict[str, Any]]) -> Optional[Tuple]:
        if string_el is None:
            return None

        style_attr = string_el.get('STYLE') or ''
        tokens = [token.strip(' ,;') for token in style_attr.split() if token]
        style_id = None
        for token in tokens:
            if token in fonts:
                style_id = token
                break

        font_entry = fonts.get(style_id, {}) if style_id else {}
        font_size = font_entry.get('font_size')
        if font_size is None:
            font_size = self._safe_float(string_el.get('FONTSIZE'))
        if font_size is None:
            font_size = self._safe_float(string_el.get('HEIGHT'))

        font_family = font_entry.get('font_family') or (string_el.get('FONTFAMILY') or '')
        font_family = self._clean_text(font_family)

        style_lower = style_attr.lower()
        weight_meta = (font_entry.get('font_weight') or '').lower()
        style_meta = (font_entry.get('font_style') or '').lower()

        bold_attr = (string_el.get('BOLD') or '').lower()
        italic_attr = (string_el.get('ITALIC') or '').lower()

        is_bold = bool(
            'bold' in style_lower
            or 'bold' in weight_meta
            or bold_attr == 'true'
        )
        is_italic = bool(
            'italic' in style_lower
            or 'italic' in style_meta
            or italic_attr == 'true'
        )

        return (
            self._round_float(font_size, 2),
            is_bold,
            is_italic,
            font_family,
            style_id or '',
        )

    def _is_probably_text_page(self, page: Dict[str, Any]) -> bool:
        page_type = (page.get('pageType') or '').lower()
        title = (page.get('title') or '').lower()

        exclusion_terms = [
            'titul', 'title', 'cover', 'obal', 'obsah', 'content', 'contents',
            'ilustr', 'illustr', 'obraz', 'image', 'map', 'fotograf', 'photo',
            'reklam', 'advert', 'příloh', 'priloh', 'appendix', 'příloha',
            'příloha', 'napis', 'blank', 'prázd', 'prazd', 'index'
        ]

        for term in exclusion_terms:
            if term in page_type or term in title:
                return False

        return True

    def _analyze_paragraphs(self, alto: str) -> Tuple[Optional[Dict[str, Any]], List[float]]:
        root = self._parse_alto_root(alto)
        if root is None:
            return None, []

        fonts = self._extract_text_styles(root)

        text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextBlock')
        if not text_blocks:
            text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextBlock')

        if not text_blocks:
            return None, []

        size_stats: Dict[float, Dict[str, Any]] = {}
        total_chars = 0
        total_paragraphs = 0
        total_lines = 0
        block_size_entries: List[Dict[str, Any]] = []
        paragraph_modes: List[float] = []

        for block_elem in text_blocks:
            style_refs = block_elem.get('STYLEREFS', '')
            tag = 'p'

            if ' ' in style_refs:
                parts = style_refs.split()
                if len(parts) > 1:
                    font_id = parts[1]
                    font_entry = fonts.get(font_id, {})
                    size = font_entry.get('font_size', 0)
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

            print(f"DEBUG: vertical_gaps={vertical_gaps}")
            print(f"DEBUG: horizontal_shifts={horizontal_shifts}")

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

            current_char_counter = Counter()
            current_bold_counter = Counter()
            current_italic_counter = Counter()
            current_family_counter = {}
            current_lines = 0
            current_total_chars = 0
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
                        # Finalize current paragraph
                        if current_total_chars > 0 and current_char_counter:
                            dominant_size = max(current_char_counter, key=current_char_counter.get)
                            dominant_chars = current_char_counter[dominant_size]
                            paragraph_modes.append(dominant_size)

                            entry = size_stats.setdefault(dominant_size, {
                                'char': 0,
                                'bold': 0,
                                'italic': 0,
                                'families': Counter(),
                                'blocks': 0,
                            })
                            entry['char'] += dominant_chars
                            entry['bold'] += current_bold_counter.get(dominant_size, 0)
                            entry['italic'] += current_italic_counter.get(dominant_size, 0)
                            if dominant_size in current_family_counter:
                                entry['families'].update(current_family_counter[dominant_size])
                            entry['blocks'] += 1

                            block_size_entries.append({
                                'size': dominant_size,
                                'charCount': dominant_chars,
                                'boldChars': current_bold_counter.get(dominant_size, 0),
                                'italicChars': current_italic_counter.get(dominant_size, 0),
                            })

                            total_chars += dominant_chars
                            total_paragraphs += 1
                            total_lines += current_lines

                        # Reset for new paragraph
                        current_char_counter = Counter()
                        current_bold_counter = Counter()
                        current_italic_counter = Counter()
                        current_family_counter = {}
                        current_lines = 0
                        current_total_chars = 0

                last_bottom = bottom

                if last_left is not None:
                    h_diff = text_line_hpos - last_left
                    if h_diff > horizontal_threshold:
                        # Finalize current paragraph
                        if current_total_chars > 0 and current_char_counter:
                            dominant_size = max(current_char_counter, key=current_char_counter.get)
                            dominant_chars = current_char_counter[dominant_size]
                            paragraph_modes.append(dominant_size)

                            entry = size_stats.setdefault(dominant_size, {
                                'char': 0,
                                'bold': 0,
                                'italic': 0,
                                'families': Counter(),
                                'blocks': 0,
                            })
                            entry['char'] += dominant_chars
                            entry['bold'] += current_bold_counter.get(dominant_size, 0)
                            entry['italic'] += current_italic_counter.get(dominant_size, 0)
                            if dominant_size in current_family_counter:
                                entry['families'].update(current_family_counter[dominant_size])
                            entry['blocks'] += 1

                            block_size_entries.append({
                                'size': dominant_size,
                                'charCount': dominant_chars,
                                'boldChars': current_bold_counter.get(dominant_size, 0),
                                'italicChars': current_italic_counter.get(dominant_size, 0),
                            })

                            total_chars += dominant_chars
                            total_paragraphs += 1
                            total_lines += current_lines

                        # Reset for new paragraph
                        current_char_counter = Counter()
                        current_bold_counter = Counter()
                        current_italic_counter = Counter()
                        current_family_counter = {}
                        current_lines = 0
                        current_total_chars = 0

                last_left = text_line_hpos

                strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not strings:
                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')

                line_all_bold = True
                line_has_content = False

                for string_el in strings:
                    style = string_el.get('STYLE', '')
                    if not style or 'bold' not in style:
                        line_all_bold = False

                    content = string_el.get('CONTENT', '')
                    subs_content = string_el.get('SUBS_CONTENT', '')
                    subs_type = string_el.get('SUBS_TYPE', '')

                    content = html.unescape(content)
                    subs_content = html.unescape(subs_content)

                    if subs_type == 'HypPart1':
                        content = subs_content
                    elif subs_type == 'HypPart2':
                        continue

                    text_value = content.strip()
                    if not text_value:
                        continue

                    char_count = len(re.sub(r'\s+', '', text_value))
                    if char_count <= 0:
                        continue

                    signature = self._string_style_signature(string_el, fonts)
                    if signature is None:
                        continue

                    font_size = signature[0]
                    if font_size is None:
                        continue

                    font_size = self._round_float(font_size, 2)
                    if font_size is None:
                        continue

                    current_char_counter[font_size] += char_count
                    current_total_chars += char_count
                    line_has_content = True

                    print(f"DEBUG string: content='{content}', style='{style}', is_bold={signature[1]}")
                    if signature[1]:  # bold
                        current_bold_counter[font_size] += char_count

                    if signature[2]:  # italic
                        current_italic_counter[font_size] += char_count

                    family = signature[3]
                    if family:
                        if font_size not in current_family_counter:
                            current_family_counter[font_size] = Counter()
                        current_family_counter[font_size][family] += char_count

                if line_has_content:
                    current_lines += 1

                line_text = ' '.join([string_el.get('CONTENT', '') for string_el in strings]).strip()
                if not line_text:
                    continue

                prospective_count = current_lines + 1

                if prospective_count == 1 and line_all_bold:
                    # Finalize single bold line as paragraph
                    if current_total_chars > 0 and current_char_counter:
                        dominant_size = max(current_char_counter, key=current_char_counter.get)
                        dominant_chars = current_char_counter[dominant_size]
                        paragraph_modes.append(dominant_size)

                        entry = size_stats.setdefault(dominant_size, {
                            'char': 0,
                            'bold': 0,
                            'italic': 0,
                            'families': Counter(),
                            'blocks': 0,
                        })
                        entry['char'] += dominant_chars
                        entry['bold'] += current_bold_counter.get(dominant_size, 0)
                        entry['italic'] += current_italic_counter.get(dominant_size, 0)
                        if dominant_size in current_family_counter:
                            entry['families'].update(current_family_counter[dominant_size])
                        entry['blocks'] += 1

                        block_size_entries.append({
                            'size': dominant_size,
                            'charCount': dominant_chars,
                            'boldChars': current_bold_counter.get(dominant_size, 0),
                            'italicChars': current_italic_counter.get(dominant_size, 0),
                        })

                        total_chars += dominant_chars
                        total_paragraphs += 1
                        total_lines += 1

                    # Reset for new paragraph
                    current_char_counter = Counter()
                    current_bold_counter = Counter()
                    current_italic_counter = Counter()
                    current_family_counter = {}
                    current_lines = 0
                    current_total_chars = 0
                    last_left = text_line_hpos
                    last_bottom = bottom
                    continue

            # Finalize last paragraph in block
            if current_total_chars > 0 and current_char_counter:
                dominant_size = max(current_char_counter, key=current_char_counter.get)
                dominant_chars = current_char_counter[dominant_size]
                paragraph_modes.append(dominant_size)

                entry = size_stats.setdefault(dominant_size, {
                    'char': 0,
                    'bold': 0,
                    'italic': 0,
                    'families': Counter(),
                    'blocks': 0,
                })
                entry['char'] += dominant_chars
                entry['bold'] += current_bold_counter.get(dominant_size, 0)
                entry['italic'] += current_italic_counter.get(dominant_size, 0)
                if dominant_size in current_family_counter:
                    entry['families'].update(current_family_counter[dominant_size])
                entry['blocks'] += 1

                block_size_entries.append({
                    'size': dominant_size,
                    'charCount': dominant_chars,
                    'boldChars': current_bold_counter.get(dominant_size, 0),
                    'italicChars': current_italic_counter.get(dominant_size, 0),
                })

                total_chars += dominant_chars
                total_paragraphs += 1
                total_lines += current_lines

        if not size_stats:
            return None, []

        analysis = {
            'size_stats': size_stats,
            'total_chars': total_chars,
            'total_blocks': total_paragraphs,
            'total_lines': total_lines,
            'block_sizes': block_size_entries,
        }

        return analysis, paragraph_modes

    def _analyze_text_blocks(self, alto: str) -> Optional[Dict[str, Any]]:
        root = self._parse_alto_root(alto)
        if root is None:
            return None

        fonts = self._extract_text_styles(root)

        text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextBlock')
        if not text_blocks:
            text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextBlock')

        if not text_blocks:
            return None

        size_stats: Dict[float, Dict[str, Any]] = {}
        total_chars = 0
        total_blocks = 0
        total_lines = 0
        block_size_entries: List[Dict[str, Any]] = []

        for block_elem in text_blocks:
            size_char_counter: Dict[float, int] = {}
            size_bold_counter: Dict[float, int] = {}
            size_italic_counter: Dict[float, int] = {}
            size_family_counter: Dict[float, Counter] = {}

            block_total_chars = 0

            text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextLine')
            if not text_lines:
                text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextLine')

            if not text_lines:
                continue

            block_line_count = 0

            for line in text_lines:
                strings = []
                for candidate in line.iter():
                    candidate_tag = getattr(candidate, 'tag', '')
                    if isinstance(candidate_tag, str) and candidate_tag.endswith('String'):
                        strings.append(candidate)

                line_has_content = False

                for string_el in strings:
                    content = string_el.get('CONTENT', '')
                    subs_content = string_el.get('SUBS_CONTENT', '')
                    subs_type = string_el.get('SUBS_TYPE', '')

                    content = html.unescape(content)
                    subs_content = html.unescape(subs_content)

                    if subs_type == 'HypPart1':
                        content = subs_content
                    elif subs_type == 'HypPart2':
                        continue

                    text_value = content.strip()
                    if not text_value:
                        continue

                    char_count = len(re.sub(r'\s+', '', text_value))
                    if char_count <= 0:
                        continue

                    signature = self._string_style_signature(string_el, fonts)
                    if signature is None:
                        continue

                    font_size_value = signature[0]
                    if font_size_value is None or not isinstance(font_size_value, (int, float)):
                        continue

                    font_size = self._round_float(font_size_value, 2)
                    if font_size is None:
                        continue

                    size_char_counter[font_size] = size_char_counter.get(font_size, 0) + char_count
                    if signature[1]:
                        size_bold_counter[font_size] = size_bold_counter.get(font_size, 0) + char_count
                    if signature[2]:
                        size_italic_counter[font_size] = size_italic_counter.get(font_size, 0) + char_count

                    family = signature[3]
                    if family:
                        fam_counter = size_family_counter.setdefault(font_size, Counter())
                        fam_counter[family] += char_count

                    block_total_chars += char_count
                    line_has_content = True

                if line_has_content:
                    block_line_count += 1

            if block_total_chars < BLOCK_MIN_TOTAL_CHARS or not size_char_counter:
                continue

            mode_size, mode_chars = max(size_char_counter.items(), key=lambda item: item[1])

            stats_entry = size_stats.setdefault(mode_size, {
                'char': 0,
                'bold': 0,
                'italic': 0,
                'families': Counter(),
                'blocks': 0,
            })

            stats_entry['char'] += mode_chars
            stats_entry['bold'] += size_bold_counter.get(mode_size, 0)
            stats_entry['italic'] += size_italic_counter.get(mode_size, 0)
            if mode_size in size_family_counter:
                stats_entry['families'].update(size_family_counter[mode_size])
            stats_entry['blocks'] += 1

            total_chars += mode_chars
            total_blocks += 1
            total_lines += block_line_count

            block_size_entries.append({
                'size': mode_size,
                'charCount': mode_chars,
                'boldChars': size_bold_counter.get(mode_size, 0),
                'italicChars': size_italic_counter.get(mode_size, 0),
            })

        if not size_stats:
            return None

        return {
            'size_stats': size_stats,
            'total_chars': total_chars,
            'total_blocks': total_blocks,
            'total_lines': total_lines,
            'block_sizes': block_size_entries,
        }

    def _compute_wave_indices(self, total_pages: int, wave_index: int) -> List[int]:
        if total_pages <= 0:
            return []

        sample_count = min(TEXT_SAMPLE_WAVE_SIZE, total_pages)
        if sample_count <= 0:
            return []

        offsets = [0.0, 0.5, 0.25]
        offset = offsets[wave_index] if wave_index < len(offsets) else 0.0
        base = sample_count + 1
        max_index = total_pages - 1

        indices: List[int] = []
        for i in range(sample_count):
            fraction = (i + 1 + offset) / base
            fraction = max(0.0, min(0.999, fraction))
            index = int(round(fraction * max_index))
            indices.append(index)

        # Zachovat pořadí a odstranit duplikáty
        seen: set[int] = set()
        unique_indices: List[int] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        return unique_indices

    def summarize_book_text_format(self, book_uuid: str, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        cache_key = book_uuid or ''
        sample_target_default = TEXT_SAMPLE_WAVE_SIZE

        if not cache_key:
            return {
                'average_height': None,
                'confidence': 0,
                'pages_used': 0,
            }

        cached = self._book_text_cache.get(cache_key)
        if cached is not None:
            return cached

        total_pages = len(pages)
        if total_pages <= 0:
            result = {
                'average_height': None,
                'confidence': 0,
                'pages_used': 0,
            }
            self._book_text_cache[cache_key] = result
            return result

        def compute_average_height_for_page(page_uuid: str) -> Optional[Dict[str, Any]]:
            print(f"[height-calc] Processing page {page_uuid}")
            alto_xml = self.get_alto_data(page_uuid)
            if not alto_xml:
                print(f"[height-calc] No ALTO XML for page {page_uuid}")
                return None
            root = self._parse_alto_root(alto_xml)
            if root is None:
                print(f"[height-calc] Failed to parse ALTO XML for page {page_uuid}")
                return None

            text_lines = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextLine')
            if not text_lines:
                text_lines = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextLine')
            print(f"[height-calc] Found {len(text_lines)} text lines for page {page_uuid}")

            word_heights = []
            for tl in text_lines:
                string_els = tl.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not string_els:
                    string_els = tl.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')
                for s in string_els:
                    h = s.get('HEIGHT')
                    if h:
                        try:
                            word_heights.append(float(h))
                        except ValueError:
                            continue
            print(f"[height-calc] Found {len(word_heights)} word heights for page {page_uuid}")

            if len(word_heights) < MIN_WORDS_PER_PAGE:
                print(f"[height-calc] Only {len(word_heights)} words, less than minimum {MIN_WORDS_PER_PAGE}, skipping page {page_uuid}")
                return None

            # Compute mode of word heights
            rounded_heights = [round(h, 1) for h in word_heights]
            height_counts = Counter(rounded_heights)
            mode_height = height_counts.most_common(1)[0][0]
            print(f"[height-calc] Mode word height for page {page_uuid}: {mode_height}")
            return {
                'mode_height': mode_height,
                'word_heights': word_heights,
                'lines_count': len(text_lines),
                'word_count': len(word_heights),
                'uuid': page_uuid
            }

        heights_per_page = []
        all_word_heights = []
        pages_used = 0
        total_lines_sampled = 0
        total_words_sampled = 0
        sampled_page_uuids = []

        # Filtrovat stránky na pravděpodobně textové
        text_pages = [p for p in pages if self._is_probably_text_page(p)]
        print(f"[text-format] Filtered to {len(text_pages)} probable text pages out of {len(pages)} total")

        if not text_pages:
            print(f"[text-format] No text pages found, cannot calculate height")
            result = {
                'average_height': None,
                'confidence': 0,
                'pages_used': 0,
            }
            self._book_text_cache[cache_key] = result
            return result

        # Použít distribuované vzorkování
        wave_index = 0
        sampled_indices = self._compute_wave_indices(len(text_pages), wave_index)
        initial_sample_count = len(sampled_indices)
        print(f"[text-format] Wave {wave_index} sampling indices: {sampled_indices}")

        for idx in sampled_indices:
            page = text_pages[idx]
            page_uuid = page.get('uuid')
            if not page_uuid:
                print(f"[text-format] Skipping page index {idx}, no UUID")
                continue
            page_data = compute_average_height_for_page(page_uuid)
            if page_data is not None:
                heights_per_page.append(page_data['mode_height'])
                all_word_heights.extend(page_data['word_heights'])
                pages_used += 1
                total_lines_sampled += page_data['lines_count']
                total_words_sampled += page_data['word_count']
                sampled_page_uuids.append(page_data['uuid'])
                print(f"[text-format] Added mode height {page_data['mode_height']} from page index {idx} (page {page.get('index', idx)})")
            else:
                print(f"[text-format] Failed to get height from page index {idx}")

        if len(heights_per_page) < 2:
            # Nedostatek dat pro výpočet variance
            print(f"[text-format] Only {len(heights_per_page)} valid pages, need at least 2 for variance calculation")
            result = {
                'average_height': None,
                'confidence': 0,
                'pages_used': pages_used,
            }
            self._book_text_cache[cache_key] = result
            return result

        # Vyhodnotit rozptyl hodnot a případně přidat další vlny
        print(f"[text-format] Heights per page: {heights_per_page}")
        wave_index = 0  # už jsme udělali wave 0
        while len(heights_per_page) > 1 and wave_index + 1 < TEXT_SAMPLE_MAX_WAVES:
            stdev = statistics.stdev(heights_per_page)
            mean = statistics.mean(heights_per_page)
            print(f"[text-format] Wave {wave_index} stdev: {stdev}, mean: {mean}")
            if stdev > 0.1 * mean:
                wave_index += 1
                additional_indices = self._compute_wave_indices(len(text_pages), wave_index)
                print(f"[text-format] Adding wave {wave_index} with indices: {additional_indices}")
                for idx in additional_indices:
                    if idx >= len(text_pages):
                        continue
                    page = text_pages[idx]
                    page_uuid = page.get('uuid')
                    if not page_uuid:
                        print(f"[text-format] Skipping additional page index {idx}, no UUID")
                        continue
                    page_data = compute_average_height_for_page(page_uuid)
                    if page_data is not None:
                        heights_per_page.append(page_data['mode_height'])
                        all_word_heights.extend(page_data['word_heights'])
                        pages_used += 1
                        total_lines_sampled += page_data['lines_count']
                        total_words_sampled += page_data['word_count']
                        sampled_page_uuids.append(page_data['uuid'])
                        print(f"[text-format] Added additional mode height {page_data['mode_height']} from page index {idx} (wave {wave_index})")
                    else:
                        print(f"[text-format] Failed to get additional height from page index {idx}")
            else:
                print(f"[text-format] Variance acceptable after wave {wave_index}, stopping")
                break
        if len(heights_per_page) <= 1:
            print(f"[text-format] Only one height or less, skipping variance check")

        if not heights_per_page:
            print(f"[text-format] No heights collected, returning None")
            result = {
                'average_height': None,
                'confidence': 0,
                'pages_used': pages_used,
            }
            self._book_text_cache[cache_key] = result
            return result

        # Compute average of all word heights after removing outliers
        if all_word_heights:
            sorted_heights = sorted(all_word_heights)
            q1 = statistics.quantiles(sorted_heights, n=4)[0]
            q3 = statistics.quantiles(sorted_heights, n=4)[2]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_heights = [h for h in all_word_heights if lower_bound <= h <= upper_bound]
            if filtered_heights:
                final_mean = round(sum(filtered_heights) / len(filtered_heights), 2)
                print(f"[text-format] Final mean height after outlier removal: {final_mean} from {len(filtered_heights)} words (out of {len(all_word_heights)} total)")
            else:
                final_mean = round(sum(all_word_heights) / len(all_word_heights), 2)
                print(f"[text-format] No outliers removed, final mean height: {final_mean} from {len(all_word_heights)} words")
        else:
            final_mean = round(sum(heights_per_page) / len(heights_per_page), 2)
            print(f"[text-format] No word heights, using page modes mean: {final_mean}")

        if len(heights_per_page) > 1:
            stdev = statistics.stdev(heights_per_page)
            rel_var = stdev / final_mean if final_mean else 1
            confidence = max(0, int(round(100 - rel_var * 100)))
            print(f"[text-format] Final stdev: {stdev}, relative variance: {rel_var}, confidence: {confidence}")
        else:
            confidence = 100
            print(f"[text-format] Single page, confidence: 100")

        result = {
            'basicTextStyle': {
                'fontSize': final_mean,
                'isBold': False,
                'isItalic': False,
                'fontFamily': '',
                'styleId': '',
            },
            'confidence': confidence,
            'sampledPages': pages_used,
            'sampleTarget': TEXT_SAMPLE_WAVE_SIZE,
            'linesSampled': total_lines_sampled,
            'distinctStyles': len(set(heights_per_page)),
            'sampledPageUuids': sampled_page_uuids,
            'totalSamples': total_words_sampled,
            'average_height': final_mean,
            'pages_used': pages_used,
        }

        try:
            log_payload = {
                'book': book_uuid,
                'average_height': final_mean,
                'confidence': confidence,
                'pages_used': pages_used,
            }
            print("[text-format] " + json.dumps(log_payload, ensure_ascii=False))
        except Exception as err:
            print(f"[text-format] Logging failed for book {book_uuid}: {err}")

        self._book_text_cache[cache_key] = result
        return result

    def get_item_json(self, uuid: str) -> Dict[str, Any]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return {}

        url = f"{self.api_base_url}/item/uuid:{normalized}"
        try:
            response = self.session.get(url, timeout=API_TIMEOUT)
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
            response = self.session.get(url, timeout=CHILDREN_TIMEOUT)
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
            response = self.session.get(url, timeout=MODS_TIMEOUT)
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
    
    def _pick_book_uuid_from_context(self, item_data: Dict[str, Any]) -> Optional[str]:
        """Z kontextové cesty vybere nejbližší 'knižní' předek.
        Preferuje nejnižší (nejbližší) výskyt `monographunit`, pak `monograph`,
        případně `periodicalitem`. Vrací UUID bez prefixu `uuid:` nebo None.
        """
        paths = item_data.get('context') or []
        if not paths:
            return None
        # context[0] bývá cesta od rootu k listu (stránce); projdeme ji odspoda
        path = paths[0] or []
        for node in reversed(path):
            model = (node.get('model') or '').lower()
            pid = node.get('pid') or ''
            if not pid:
                continue
            if model in ('monographunit', 'monograph', 'periodicalitem'):
                return self._strip_uuid_prefix(pid)
        return None

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
            # Nejprve vezmeme nejbližší knižní předek z context path (svazek/monografie/číslo)
            book_uuid = self._pick_book_uuid_from_context(item_data) or ''
            # Fallback: root_pid (může ukazovat na sérii; proto až druhý pokus)
            if not book_uuid:
                root_pid = item_data.get('root_pid') or ''
                book_uuid = self._strip_uuid_prefix(root_pid)
        else:
            # Když není stránka a je to rovnou kniha/svazek/číslo, použij ji přímo
            if (model or '').lower() in ('monographunit', 'monograph', 'periodicalitem'):
                book_uuid = self._strip_uuid_prefix(item_data.get('pid'))
            else:
                # Jiný uzel – zkusíme najít nejbližší knižní předek z contextu
                book_uuid = self._pick_book_uuid_from_context(item_data) or self._strip_uuid_prefix(item_data.get('pid'))

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

        book_constants = self.summarize_book_text_format(book_uuid, pages)

        return {
            "book_uuid": book_uuid,
            "book": book_data,
            "page_uuid": page_uuid,
            "page": page_summary,
            "page_item": page_data,
            "pages": pages,
            "mods": self.get_mods_metadata(book_uuid),
            "current_index": resolved_index,
            "book_constants": book_constants,
        }

    def get_alto_data(self, uuid: str) -> str:
        """Stáhne ALTO XML data pro daný UUID"""
        url = f"https://kramerius5.nkp.cz/search/api/v5.0/item/uuid:{uuid}/streams/ALTO"
        try:
            response = self.session.get(url, timeout=ALTO_TIMEOUT)
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
        """Získá bloky textu pro čtení (sestavené z TextLine podle vertikálních mezer)."""
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

        if page is None or print_space is None:
            return []

        alto_height = int(page.get('HEIGHT', '0'))
        alto_width = int(page.get('WIDTH', '0'))
        alto_height2 = int(print_space.get('HEIGHT', '0'))
        alto_width2 = int(print_space.get('WIDTH', '0'))

        aw = alto_width if (alto_height > 0 and alto_width > 0) else alto_width2
        ah = alto_height if (alto_height > 0 and alto_width > 0) else alto_height2

        # Posbírej řádky
        text_lines = [el for el in root.iter() if getattr(el, 'tag', '').endswith('TextLine')]

        blocks: List[Dict[str, Any]] = []
        block = {'text': '', 'hMin': 0, 'hMax': 0, 'vMin': 0, 'vMax': 0, 'width': aw, 'height': ah}

        last_bottom = 0
        for text_line in text_lines:
            # ignoruj velmi krátké pseudo-řádky
            text_line_width = int(text_line.get('WIDTH', '0') or 0)
            if text_line_width < 50:
                continue

            hpos = int(text_line.get('HPOS', '0') or 0)
            vpos = int(text_line.get('VPOS', '0') or 0)
            height = int(text_line.get('HEIGHT', '0') or 0)
            bottom = vpos + height

            # Nový odstavec, když je větší mezera mezi předchozím a aktuálním řádkem
            if last_bottom > 0 and (vpos - last_bottom) > 50:
                if block['text'].strip():
                    blocks.append(block)
                block = {'text': '', 'hMin': 0, 'hMax': 0, 'vMin': 0, 'vMax': 0, 'width': aw, 'height': ah}

            # Přidání textu řádku
            line_parts: List[str] = []
            for elem in text_line.iter():
                if getattr(elem, 'tag', '').endswith('String'):
                    content = elem.get('CONTENT', '') or ''
                    subs_content = elem.get('SUBS_CONTENT', '') or ''
                    subs_type = elem.get('SUBS_TYPE', '') or ''
                    if subs_type == 'HypPart1':
                        content = subs_content
                    elif subs_type == 'HypPart2':
                        continue
                    line_parts.append(content)

            line_text = ' '.join(p for p in line_parts if p).strip()
            if line_text:
                if not block['text']:
                    block['hMin'] = hpos
                    block['vMin'] = vpos
                block['text'] += line_text + '\n'
                block['hMax'] = max(block['hMax'], hpos + text_line_width)
                block['vMax'] = max(block['vMax'], bottom)

            last_bottom = bottom

        # poslední blok
        if block['text'].strip():
            blocks.append(block)

        return blocks

    def get_formatted_text(self, alto: str, uuid: str, width: int, height: int, average_height: Optional[float] = None) -> str:
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

        def finalize_block(block_data, word_heights, average_height, heading_fonts):
            """Po spojení řádků uloží blok, pokud není prázdný."""

            text_content = ' '.join(block_data['lines']).strip()
            if not text_content:
                return

            tag = block_data['tag']

            if tag != 'h3' and average_height is not None and word_heights:
                max_font = max(block_data['font_sizes']) if block_data['font_sizes'] else 0

                if max_font in heading_fonts:
                    min_ratio = HEADING_MIN_WORD_RATIO_DEFAULT * HEADING_FONT_RATIO_MULTIPLIER
                else:
                    min_ratio = HEADING_MIN_WORD_RATIO_DEFAULT

                count_above_h1 = sum(1 for h in word_heights if h >= average_height * HEADING_H1_RATIO)
                count_above_h2 = sum(1 for h in word_heights if h >= average_height * HEADING_H2_RATIO)
                total_words = len(word_heights)

                if total_words > 0:
                    if count_above_h1 / total_words >= min_ratio:
                        tag = 'h1'
                        print(f"DEBUG finalize_block: changed to h1, count_above_h1={count_above_h1}, total_words={total_words}, ratio={count_above_h1 / total_words:.3f} >= {min_ratio:.3f}")
                    elif count_above_h2 / total_words >= min_ratio:
                        tag = 'h2'
                        print(f"DEBUG finalize_block: changed to h2, count_above_h2={count_above_h2}, total_words={total_words}, ratio={count_above_h2 / total_words:.3f} >= {min_ratio:.3f}")
                    else:
                        tag = 'p'
                        print(f"DEBUG finalize_block: stayed p, count_above_h1={count_above_h1}, count_above_h2={count_above_h2}, total_words={total_words}, min_ratio={min_ratio:.3f}")
                else:
                    tag = 'p'

                # Debug print for block details
                print(f"DEBUG finalize_block: text='{text_content[:100]}...', tag={tag}, max_font={max_font}, average_height={average_height}, word_heights={word_heights}, count_above_h1={count_above_h1}, count_above_h2={count_above_h2}, total_words={total_words}, min_ratio={min_ratio:.3f}")

            blocks.append({'text': text_content, 'tag': tag, 'centered': block_data.get('centered', False), 'all_bold': block_data.get('all_bold', False)})

        # Pokud average_height není poskytnut, načteme z kontextu knihy
        if average_height is None and uuid:
            context = self.get_book_context(uuid)
            average_height = context.get('book_constants', {}).get('average_height') if context else None

        # Zpracování TextBlocků
        text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextBlock')
        if not text_blocks:
            text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextBlock')

        print(f"DEBUG: Found {len(text_blocks)} TextBlocks")

        # Pre-scan all text lines to build font_counts based on characters
        font_counts = defaultdict(int)
        for block_elem in text_blocks:
            text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextLine')
            if not text_lines:
                text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextLine')
            for text_line in text_lines:
                text_line_width = int(text_line.get('WIDTH', '0'))
                if text_line_width < 50:
                    continue
                style_ref = text_line.get('STYLEREFS') or block_elem.get('STYLEREFS', '')
                current_line_font_size = 0
                if style_ref:
                    parts = style_ref.split()
                    if len(parts) > 1:
                        font_id = parts[1]
                        current_line_font_size = fonts.get(font_id, 0)
                strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not strings:
                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')
                total_char_in_line = 0
                for string_el in strings:
                    content = string_el.get('CONTENT', '')
                    subs_content = string_el.get('SUBS_CONTENT', '')
                    subs_type = string_el.get('SUBS_TYPE', '')
                    content = html.unescape(content)
                    subs_content = html.unescape(subs_content)
                    if subs_type == 'HypPart1':
                        content = subs_content
                    elif subs_type == 'HypPart2':
                        continue
                    text_value = content.strip()
                    if text_value:
                        char_count = len(re.sub(r'\s+', '', text_value))
                        total_char_in_line += char_count
                font_counts[current_line_font_size] += total_char_in_line

        # Identifikace nadpisových fontů na základě rozdílů ve velikostech a zastoupení
        sorted_sizes = sorted(set(fonts.values()), reverse=True)
        heading_fonts = []
        boundary_found = False
        for i in range(len(sorted_sizes) - 1):
            if sorted_sizes[i] > sorted_sizes[i + 1] * HEADING_FONT_GAP_THRESHOLD:
                # Found boundary, collect all sizes above it
                candidate_sizes = sorted_sizes[:i+1]
                # Check combined ratio
                total_candidate_chars = sum(font_counts.get(size, 0) for size in candidate_sizes)
                total_chars = sum(font_counts.values())
                if total_chars > 0:
                    combined_ratio = total_candidate_chars / total_chars
                    if combined_ratio <= HEADING_FONT_MAX_RATIO:
                        heading_fonts.extend(candidate_sizes)
                boundary_found = True
                break
        if not boundary_found:
            # No boundary found, check if the largest font is rare enough
            if sorted_sizes:
                largest_size = sorted_sizes[0]
                largest_chars = font_counts.get(largest_size, 0)
                total_chars = sum(font_counts.values())
                if total_chars > 0 and largest_chars / total_chars <= HEADING_FONT_MAX_RATIO:
                    heading_fonts.append(largest_size)

        print(f"DEBUG: heading_fonts={heading_fonts}, font_counts={dict(font_counts)}")

        for block_elem in text_blocks:
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

            current_block = {'lines': [], 'tag': 'p', 'font_sizes': set(), 'word_heights': [], 'all_bold': True}

            line_heights = [record['height'] for record in line_records]
            line_widths = [record['width'] for record in line_records]

            vertical_gaps = []
            horizontal_shifts = []
            previous_bottom = None
            previous_left = None

            # Calculate line centers for center alignment detection
            line_centers = [record['hpos'] + record['width'] / 2 for record in line_records]
            median_center = statistics.median(line_centers) if line_centers else 0
            median_width = statistics.median(line_widths) if line_widths else 0
            margin = median_width * CENTER_ALIGNMENT_ERROR_MARGIN if median_width else 0

            # Check if all line centers are within median_center ± margin
            centers_aligned = all(
                abs(center - median_center) <= margin for center in line_centers
            )

            # Check if line widths vary enough (not a justified block)
            if line_widths:
                min_width = min(line_widths)
                max_width = max(line_widths)
                width_diff = max_width - min_width
                width_diff_threshold = median_width * CENTER_ALIGNMENT_MIN_LINE_LEN_DIFF if median_width else 0
                widths_vary = width_diff > width_diff_threshold
            else:
                widths_vary = False

            is_center_aligned = centers_aligned and widths_vary

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

            print(f"DEBUG: line_heights={line_heights}")
            print(f"DEBUG: line_widths={line_widths}")
            print(f"DEBUG: vertical_gaps={vertical_gaps}, horizontal_shifts={horizontal_shifts}")
            print(f"DEBUG: median_height={median_height}, median_width={median_width}, median_gap={median_gap}, median_shift={median_shift}")
            print(f"DEBUG: trimmed_shifts={trimmed_shifts}")
            print(f"DEBUG: vertical_threshold_candidates={vertical_threshold_candidates}, horizontal_threshold_candidates={horizontal_threshold_candidates}")
            print(f"DEBUG: vertical_threshold={vertical_threshold}, horizontal_threshold={horizontal_threshold}")
            print(f"DEBUG: heading_fonts={heading_fonts}, font_counts={dict(font_counts)}")
            print(f"DEBUG: is_center_aligned={is_center_aligned}")

            tag = 'p'
            current_block = {'lines': [], 'tag': tag, 'font_sizes': set(), 'word_heights': [], 'centered': is_center_aligned, 'all_bold': True}
            lines = 0
            last_bottom = None
            last_left = None

            for record in line_records:
                text_line = record['element']
                text_line_vpos = record['vpos']
                text_line_hpos = record['hpos']
                bottom = record['bottom']

                # Extract font size for current line from STYLEREFS on TextLine or inherit from TextBlock
                style_ref = text_line.get('STYLEREFS') or block_elem.get('STYLEREFS', '')
                current_line_font_size = 0
                if style_ref:
                    parts = style_ref.split()
                    if len(parts) > 1:
                        font_id = parts[1]
                        current_line_font_size = fonts.get(font_id, 0)

                if last_bottom is not None:
                    v_diff = text_line_vpos - last_bottom
                    print(f"DEBUG: v_diff={v_diff}, vertical_threshold={vertical_threshold}")
                    if v_diff > vertical_threshold:
                        print(f"DEBUG: Splitting on v_diff={v_diff} > {vertical_threshold}")
                        # Velká vertikální mezera = nový blok textu
                        if current_block['lines']:
                            finalize_block(current_block, current_block['word_heights'], average_height, heading_fonts)
                        current_block = {'lines': [], 'tag': tag, 'font_sizes': set(), 'word_heights': [], 'centered': is_center_aligned, 'all_bold': True}
                        lines = 0

                last_bottom = bottom

                if last_left is not None and not is_center_aligned:
                    h_diff = text_line_hpos - last_left
                    # Check font size difference threshold (e.g., 1.2x)
                    font_size_differs = False
                    if last_line_font_size and current_line_font_size:
                        ratio = max(last_line_font_size, current_line_font_size) / min(last_line_font_size, current_line_font_size)
                        if ratio >= 1.2:
                            font_size_differs = True

                    print(f"DEBUG: h_diff={h_diff}, horizontal_threshold={horizontal_threshold}, font_size_differs={font_size_differs}")

                    if h_diff > horizontal_threshold or font_size_differs:
                        print(f"DEBUG: Horizontal split on h_diff={h_diff} > {horizontal_threshold} or font_size_differs={font_size_differs}")
                        # Podobně reagujeme na výrazný horizontální posun nebo rozdíl fontu (např. tabulky, sloupce, nadpisy)
                        if current_block['lines']:
                            finalize_block(current_block, current_block['word_heights'], average_height, heading_fonts)
                        current_block = {'lines': [], 'tag': tag, 'font_sizes': set(), 'word_heights': [], 'centered': is_center_aligned, 'all_bold': True}
                        lines = 0
                    elif h_diff < 0 and abs(h_diff) > horizontal_threshold * NEGATIVE_SHIFT_MULTIPLIER and current_block['lines']:
                        print(f"DEBUG: Negative split on h_diff={h_diff}, abs(h_diff)={abs(h_diff)} > {horizontal_threshold * NEGATIVE_SHIFT_MULTIPLIER}")
                        # Negative horizontal shift indicates indented line, so previous lines were single-line paragraphs that should be split
                        if len(current_block['lines']) > 1:
                            # Finalize each previous line as a separate paragraph
                            for previous_line in current_block['lines'][:-1]:
                                finalize_block({'lines': [previous_line], 'tag': tag, 'font_sizes': set(), 'word_heights': []}, [], average_height, heading_fonts)
                            # Reset current block to start with the last line
                            current_block = {'lines': [current_block['lines'][-1]], 'tag': tag, 'font_sizes': set(), 'word_heights': [], 'centered': is_center_aligned, 'all_bold': True}
                            lines = 1

                last_left = text_line_hpos
                last_line_font_size = current_line_font_size

                strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not strings:
                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')

                line_text = ' '.join([string_el.get('CONTENT', '') for string_el in strings]).strip()
                print(f"DEBUG: Processing line at hpos={text_line_hpos}, previous_left={last_left}, line_text='{line_text[:50]}...'")

                line_parts = []
                line_all_bold = True

                for string_el in strings:
                    style = string_el.get('STYLE', '')
                    print(f"DEBUG line_all_bold check: style='{style}', 'bold' in style={ 'bold' in style}")
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

                    height = string_el.get('HEIGHT')
                    if height:
                        current_block['word_heights'].append(float(height))

                    # Use the line's font size for all strings in the line
                    if current_line_font_size:
                        current_block['font_sizes'].add(current_line_font_size)

                    if content:
                        line_parts.append(content)

                if not line_all_bold:
                    current_block['all_bold'] = False

                line_text = ' '.join(line_parts).strip()
                if not line_text:
                    continue

                prospective_count = lines + 1

                current_block['lines'].append(line_text)
                lines = len(current_block['lines'])

            if current_block['lines']:
                finalize_block(current_block, current_block['word_heights'], average_height, heading_fonts)

        # After all blocks are finalized, apply the new logic for h3 detection based on bold paragraphs and neighbors
        for i, block in enumerate(blocks):
            print(f"DEBUG post-processing: block '{block['text'][:50]}...', tag={block['tag']}, all_bold={block.get('all_bold', False)}")
            if block['tag'] == 'p' and block.get('all_bold', False):
                prev_block = blocks[i - 1] if i > 0 else None
                next_block = blocks[i + 1] if i < len(blocks) - 1 else None

                def is_heading_or_nonbold_p(b):
                    if b is None:
                        return True
                    if b['tag'] in ('h1', 'h2'):
                        return True
                    if b['tag'] == 'p' and not b.get('all_bold', False):
                        return True
                    return False

                if is_heading_or_nonbold_p(prev_block) and is_heading_or_nonbold_p(next_block) and not (prev_block is None and next_block is None):
                    print(f"DEBUG post-processing: changing block '{block['text'][:50]}...' to h3")
                    block['tag'] = 'h3'

        # Generování HTML - přesně jako v TypeScript
        result = ""
        for block in blocks:
            if block['text'].strip():  # Jen neprázdné bloky
                style_attr = ''
                if block.get('centered'):
                    style_attr = ' style="text-align: center;"'
                result += f"<{block['tag']}{style_attr}>{block['text'].strip()}</{block['tag']}>"

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
