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
HORIZONTAL_WIDTH_RATIO = 0.03  # Kandidát prahu = medián šířek řádků * tato hodnota (nižší = citlivější). [<0.041]
HORIZONTAL_SHIFT_MULTIPLIER = 0.85  # Kandidát prahu = medián kladných posunů * tato hodnota.
HORIZONTAL_MIN_THRESHOLD = 12   # Minimální povolený práh pro horizontální dělení (v ALTO jednotkách).
NEGATIVE_SHIFT_MULTIPLIER = 0.95  # Negativní hranice = horizontální práh * tato hodnota (víc citlivá než pozitivní).
FALLBACK_THRESHOLD = 40        # Záložní hodnota, pokud nelze heuristiku spočítat.

CENTER_ALIGNMENT_ERROR_MARGIN = 0.02  # 5% margin for center alignment detection relative to median line width
CENTER_ALIGNMENT_MIN_LINE_LEN_DIFF = 0.1  # 10% minimum difference between shortest and longest line for center alignment
FONT_SIZE_SPLIT_RATIO = 1.2  # Ratio difference between consecutive line font sizes that triggers a split
CENTER_LINE_HEIGHT_RATIO = 1.25  # Ratio threshold for average word heights when splitting centered blocks
SINGLE_LINE_VERTICAL_GAP_RATIO = 3.0  # Tolerated gap multiplier for re-centering single-line block sequences

# Konstanty pro rozhodování o nadpisech na základě výšky slov
HEADING_H2_RATIO = 1.08                 # Práh pro h2: 1.08 * průměrná výška
HEADING_H1_RATIO = 2                    # Práh pro h1: 1.6 * průměrná výška
HEADING_MIN_WORD_RATIO_DEFAULT = 0.82   # Výchozí minimální podíl slov v bloku, které musí překročit práh (margin of error pro OCR) [> 0.819]
HEADING_FONT_GAP_THRESHOLD = 1.2        # Práh pro rozdíl ve velikosti fontu pro identifikaci nadpisových fontů
HEADING_FONT_RATIO_MULTIPLIER = 0.56    # Koeficient pro snížení prahu pro bloky s nadpisovými fonty [>0.55]
HEADING_FONT_MAX_RATIO = 0.4            # Maximální podíl řádků s fontem, aby byl považován za nadpisový
HEADING_FONT_MERGE_TOLERANCE = 0.1      # Povolená relativní odchylka mezi velikostmi písma při spojování nadpisů

# Konstanty pro rozhodování o "malém" textu (poznámky pod čarou, popisky obrázků, atd.)
SMALL_RATIO = 0.92                      # Práh pro "malý" text: 0.92 * průměrná výška
SMALL_MIN_WORD_RATIO = 0.61             # Výchozí minimální podíl slov v bloku, které musí překročit práh "malého" textu (např. poznámky pod čarou) [>0.6]
SMALL_RATIO_MULTIPLIER = 0.7            # Koeficient pro snížení prahu pro malé bloky (např. při detekci h3 nebo "*" porefix")

HYPHEN_LIKE_CHARS = "-–—‑‒−‐"           # Znaky, které vyhodnocujeme jako spojovník při rozdělených slovech

# Znaky, které považujeme za možné "noise" na začátku řádku (OCR tečky, závorky apod.)
NOISE_LEADING_PUNCT = set('.:,;)]"\'“”’')
# Kolik prvních tokenů prohledáme při hledání "effective" levého hpos
EFFECTIVE_LEFT_MAX_TOKEN_SCAN = 3

WORD_LENGTH_FILTER_INITIAL = 1          # Počáteční délka (včetně) pro filtr krátkých slov; iterativně snižujeme
WORD_LENGTH_FILTER_MIN_WORDS = 5        # Filtrovaný seznam musí mít více než tolik slov, jinak použijeme kompletní data

PAGE_NUMBER_SHORT_LINE_RATIO_TO_PAGE = 0.2  # Procento šířky stránky, pod které je řádek považován za potenciální řádek s číslem stránky
PAGE_NUMBER_ALPHA_REJECTION_RATIO = 0.5     # Pokud podíl písmen přesáhne tento poměr, řádek vyloučíme jako kandidáta
PAGE_NUMBER_MIN_NONSPACE_FOR_REJECTION = 5  # Krátké řádky (méně znaků) ponecháme i při vysokém podílu písmen
PAGE_NOTE_STYLE_ATTR = ' style="display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;"'

# Centralized HTTP timeouts (seconds)
API_TIMEOUT = 25
CHILDREN_TIMEOUT = 25
MODS_TIMEOUT = 25
ALTO_TIMEOUT = 30

# Nastavení pro analýzu typického formátu základního textu v rámci knihy.
TEXT_SAMPLE_WAVE_SIZE = 10           # Kolik stran z každé vlny odečíst.
TEXT_SAMPLE_MAX_WAVES = 5           # Maximální počet vln načítání.
# Minimální požadovaná "confidence" (jako zlomek, např. 0.9 = 90%). Pokud odhadovaná
# confidence (v rozsahu 0.0-1.0) dosáhne této hodnoty, přestaneme přidávat další vlny.
MIN_CONFIDENCE_FOR_EARLY_STOP = 0.9
BLOCK_MIN_TOTAL_CHARS = 40          # Minimální množství znaků v TextBlocku, aby šel do analýzy.
MIN_WORDS_PER_PAGE = 20             # Minimální počet slov na stránce pro výpočet průměrné výšky.

BOOK_TEXT_STYLE_CACHE: Dict[str, Dict[str, Any]] = {}

DEFAULT_API_BASES: List[str] = [
    "https://kramerius.mzk.cz/search/api/v5.0",
    "https://kramerius5.nkp.cz/search/api/v5.0",
]

class AltoProcessor:
    def __init__(
        self,
        iiif_base_url: Optional[str] = None,
        api_base_url: Optional[str] = None,
        fallback_api_bases: Optional[List[str]] = None,
    ):
        primary_iiif = iiif_base_url or "https://kramerius.mzk.cz/search/iiif"
        self.iiif_base_url = primary_iiif.rstrip('/')

        base_candidates: List[str] = []
        if api_base_url:
            base_candidates.append(api_base_url)
        if fallback_api_bases:
            base_candidates.extend(fallback_api_bases)
        base_candidates.extend(DEFAULT_API_BASES)

        self._api_base_candidates = self._normalize_api_bases(base_candidates)
        self.api_base_url = self._api_base_candidates[0] if self._api_base_candidates else ""
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
    def _normalize_api_base(value: str) -> Optional[str]:
        if not value:
            return None
        normalized = value.rstrip('/')
        return normalized or None

    def _normalize_api_bases(self, bases: List[str]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for base in bases:
            normalized = self._normalize_api_base(base)
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
        return ordered

    def _iter_api_bases(self, override: Optional[str] = None):
        seen: set[str] = set()
        if override:
            normalized = self._normalize_api_base(override)
            if normalized:
                seen.add(normalized)
                yield normalized
        for base in self._api_base_candidates:
            if base in seen:
                continue
            seen.add(base)
            yield base

    def _remember_successful_base(self, base: str) -> None:
        normalized = self._normalize_api_base(base)
        if not normalized:
            return
        try:
            self._api_base_candidates.remove(normalized)
        except ValueError:
            pass
        self._api_base_candidates.insert(0, normalized)
        self.api_base_url = normalized

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
    def _normalize_page_number(value: Optional[str]) -> Tuple[Optional[str], bool]:
        if value is None:
            return None, False

        cleaned = value.strip()
        if not cleaned:
            return None, False

        bracket_match = re.fullmatch(r"\[\s*([IVXLCDMivxlcdm]+|\d+)\s*\]", cleaned)
        if bracket_match:
            inner = bracket_match.group(1)
            if re.fullmatch(r"\d+", inner):
                return inner, False
            if re.fullmatch(r"[IVXLCDMivxlcdm]+", inner):
                return inner.upper(), False
            return inner.strip(), False

        if re.fullmatch(r"\d+", cleaned):
            return cleaned, True

        if re.fullmatch(r"[IVXLCDMivxlcdm]+", cleaned):
            return cleaned.upper(), True

        return None, False

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

            # local set for page-number detection (not used here but kept for compatibility with flows)
            page_number_line_ids: set[int] = set()
            line_records = []
            for text_line in text_lines:
                if page_number_line_ids and id(text_line) in page_number_line_ids:
                    continue

                text_line_width = int(text_line.get('WIDTH', '0') or 0)
                if text_line_width <= 0:
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
                line_width = record['width']
                line_center = text_line_hpos + line_width / 2 if line_width else float(text_line_hpos)

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

                # Collect token-level HPOS/CONTENT so we can compute an effective left
                current_line_token_hpos: List[Optional[int]] = []
                current_line_token_texts: List[str] = []
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

            # Estimate confidence from relative variance and allow early stop.
            # Use fractional confidence in range 0.0..1.0 (1 - relative_variance).
            rel_var_wave = (stdev / mean) if mean else 1.0
            estimated_confidence_frac = max(0.0, min(1.0, 1.0 - rel_var_wave))
            estimated_confidence_percent = int(round(estimated_confidence_frac * 100))
            print(f"[text-format] Wave {wave_index} estimated confidence: {estimated_confidence_frac:.3f} ({estimated_confidence_percent}%)")
            if estimated_confidence_frac >= MIN_CONFIDENCE_FOR_EARLY_STOP:
                print(f"[text-format] Confidence {estimated_confidence_percent}% >= MIN_CONFIDENCE_FOR_EARLY_STOP ({MIN_CONFIDENCE_FOR_EARLY_STOP:.2f}), stopping early")
                break

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
            rel_var = stdev / final_mean if final_mean else 1.0
            confidence_frac = max(0.0, min(1.0, 1.0 - rel_var))
            confidence = int(round(confidence_frac * 100))
            print(f"[text-format] Final stdev: {stdev}, relative variance: {rel_var:.3f}, confidence: {confidence_frac:.3f} ({confidence}%)")
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

    def get_item_json(self, uuid: str, api_base_override: Optional[str] = None) -> Dict[str, Any]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return {}

        attempted: List[str] = []
        last_error: Optional[Exception] = None
        for base in self._iter_api_bases(api_base_override):
            attempted.append(base)
            url = f"{base}/item/uuid:{normalized}"
            try:
                response = self.session.get(url, timeout=API_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                self._remember_successful_base(base)
                return data if isinstance(data, dict) else {}
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            print(f"Chyba při načítání metadat objektu {uuid} z {', '.join(attempted)}: {last_error}")
        return {}

    def get_children(self, uuid: str, api_base_override: Optional[str] = None) -> List[Dict[str, Any]]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return []

        attempted: List[str] = []
        last_error: Optional[Exception] = None
        for base in self._iter_api_bases(api_base_override):
            attempted.append(base)
            url = f"{base}/item/uuid:{normalized}/children"
            try:
                response = self.session.get(url, timeout=CHILDREN_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                self._remember_successful_base(base)
                return data if isinstance(data, list) else []
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            print(f"Chyba při načítání potomků objektu {uuid} z {', '.join(attempted)}: {last_error}")
        return []

    def get_mods_metadata(self, uuid: str, api_base_override: Optional[str] = None) -> List[Dict[str, str]]:
        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return []

        attempted: List[str] = []
        last_error: Optional[Exception] = None
        response_content: Optional[bytes] = None
        for base in self._iter_api_bases(api_base_override):
            attempted.append(base)
            url = f"{base}/item/uuid:{normalized}/streams/BIBLIO_MODS"
            try:
                response = self.session.get(url, timeout=MODS_TIMEOUT)
                response.raise_for_status()
                response_content = response.content
                self._remember_successful_base(base)
                break
            except Exception as exc:
                last_error = exc
                continue

        if response_content is None:
            if last_error:
                print(f"Chyba při načítání MODS metadat {uuid} z {', '.join(attempted)}: {last_error}")
            return []

        try:
            root = ET.fromstring(response_content)
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
            "api_base": self.api_base_url,
        }

    def get_alto_data(self, uuid: str, api_base_override: Optional[str] = None) -> str:
        """Stáhne ALTO XML data pro daný UUID"""

        normalized = self._strip_uuid_prefix(uuid)
        if not normalized:
            return ""

        attempted: List[str] = []
        last_error: Optional[Exception] = None
        for base in self._iter_api_bases(api_base_override):
            attempted.append(base)
            url = f"{base}/item/uuid:{normalized}/streams/ALTO"
            try:
                response = self.session.get(url, timeout=ALTO_TIMEOUT)
                response.raise_for_status()
                self._remember_successful_base(base)
                return response.content.decode('utf-8', errors='replace')
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            print(f"Chyba při stahování ALTO dat {uuid} z {', '.join(attempted)}: {last_error}")
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

        context = self.get_book_context(uuid) if uuid else None
        if average_height is None and context:
            average_height = context.get('book_constants', {}).get('average_height')

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

        def merge_hyphenated_line_breaks(block_data: Dict[str, Any]) -> None:
            lines: List[str] = block_data.get('lines', [])
            tokens_by_line: List[List[str]] = block_data.get('line_word_tokens', [])
            lengths_by_line: List[List[int]] = block_data.get('line_word_lengths', [])
            heights_by_line: List[List[float]] = block_data.get('line_word_heights', [])

            if not lines or not tokens_by_line:
                return

            line_count = min(len(lines), len(tokens_by_line))
            index = 0

            while index < line_count - 1:
                current_tokens = tokens_by_line[index] if index < len(tokens_by_line) else []
                next_tokens = tokens_by_line[index + 1] if index + 1 < len(tokens_by_line) else []
                if not current_tokens or not next_tokens:
                    index += 1
                    continue

                last_token = current_tokens[-1]
                next_token = next_tokens[0]
                if not last_token or not next_token:
                    index += 1
                    continue

                last_char = last_token[-1]
                if last_char not in HYPHEN_LIKE_CHARS:
                    index += 1
                    continue

                prefix = last_token[:-1]
                if not prefix or not any(ch.isalpha() for ch in prefix):
                    index += 1
                    continue

                merged_token = prefix + next_token
                current_tokens[-1] = merged_token
                next_tokens.pop(0)

                if index < len(lines):
                    lines[index] = ' '.join(current_tokens).strip()
                if index + 1 < len(lines):
                    lines[index + 1] = ' '.join(next_tokens).strip()

                if index < len(lengths_by_line):
                    current_lengths = lengths_by_line[index]
                else:
                    current_lengths = None
                if index + 1 < len(lengths_by_line):
                    next_lengths = lengths_by_line[index + 1]
                else:
                    next_lengths = None

                if current_lengths is not None and current_lengths:
                    merged_length = sum(1 for ch in merged_token if not ch.isspace())
                    current_lengths[-1] = merged_length
                if next_lengths is not None and next_lengths:
                    next_lengths.pop(0)

                if index < len(heights_by_line):
                    current_heights = heights_by_line[index]
                else:
                    current_heights = None
                if index + 1 < len(heights_by_line):
                    next_heights = heights_by_line[index + 1]
                else:
                    next_heights = None

                if current_heights is not None and current_heights:
                    extra_height = None
                    if next_heights:
                        extra_height = next_heights[0]
                    if extra_height is not None:
                        current_heights[-1] = max(current_heights[-1], extra_height)
                if next_heights is not None and next_heights:
                    next_heights.pop(0)

                print(
                    f"DEBUG hyphen_merge: merged '{last_token}' + '{next_token}' -> '{merged_token}'"
                )

                line_count = min(len(lines), len(tokens_by_line))
                index += 1

        def rebuild_block_word_metrics(block_data: Dict[str, Any]) -> None:
            aggregated_heights: List[float] = []
            aggregated_lengths: List[int] = []
            aggregated_tokens: List[str] = []

            line_heights = block_data.get('line_word_heights', []) or []
            line_lengths = block_data.get('line_word_lengths', []) or []
            line_tokens = block_data.get('line_word_tokens', []) or []

            max_lines = max(len(line_heights), len(line_lengths), len(line_tokens), len(block_data.get('lines', [])))
            for idx in range(max_lines):
                heights = line_heights[idx] if idx < len(line_heights) else []
                lengths = line_lengths[idx] if idx < len(line_lengths) else []
                tokens = line_tokens[idx] if idx < len(line_tokens) else []

                aggregated_heights.extend(heights)
                aggregated_lengths.extend(lengths)
                aggregated_tokens.extend(tokens)

            block_data['word_heights'] = aggregated_heights
            block_data['word_lengths'] = aggregated_lengths
            block_data['word_tokens'] = aggregated_tokens

        def finalize_block(block_data, word_heights, word_lengths, word_tokens, average_height, heading_fonts):
            """Po spojení řádků uloží blok, pokud není prázdný."""

            merge_hyphenated_line_breaks(block_data)
            rebuild_block_word_metrics(block_data)

            lines_for_text = [line for line in block_data.get('lines', []) if line]
            text_content = ' '.join(lines_for_text).strip()
            if not text_content:
                text_content = ' '.join(block_data.get('lines', [])).strip()
            if not text_content:
                return

            tag = block_data['tag']

            effective_word_heights = block_data.get('word_heights', [])
            effective_word_lengths = block_data.get('word_lengths', [])
            effective_word_tokens = block_data.get('word_tokens', [])
            line_font_sizes = list(block_data.get('line_font_sizes', []))
            line_bold_flags = list(block_data.get('line_bold_flags', []))
            source_block_id = block_data.get('source_block_id')
            block_all_bold = block_data.get('all_bold', False)

            if tag != 'h3' and average_height is not None and effective_word_heights:
                max_font = max(block_data['font_sizes']) if block_data['font_sizes'] else 0

                if max_font in heading_fonts:
                    min_ratio = HEADING_MIN_WORD_RATIO_DEFAULT * HEADING_FONT_RATIO_MULTIPLIER
                else:
                    min_ratio = HEADING_MIN_WORD_RATIO_DEFAULT

                # Start from triples so we can consistently filter punctuation-only tokens
                pairs = list(zip(effective_word_heights, effective_word_lengths, effective_word_tokens))
                # Remove punctuation-only tokens (no alnum chars)
                punct_removed = [token for _, _, token in pairs if not any(ch.isalnum() for ch in (token or ""))]
                if punct_removed:
                    print(f"DEBUG finalize_block: removing punctuation-only tokens before ratio: {punct_removed}")
                pairs = [p for p in pairs if any(ch.isalnum() for ch in (p[2] or ""))]

                heights_for_ratio = [h for h, _, _ in pairs]
                applied_length_filter = None

                # Apply same iterative length-filter logic as used elsewhere: try to ignore very short tokens
                if pairs and WORD_LENGTH_FILTER_INITIAL > 0:
                    total_pairs = len(pairs)
                    print(
                        "DEBUG finalize_block: length-filter precheck total_pairs=%s initial_threshold=%s"
                        % (total_pairs, WORD_LENGTH_FILTER_INITIAL)
                    )
                    for length_threshold in range(WORD_LENGTH_FILTER_INITIAL, 0, -1):
                        filtered = [h for h, length, _ in pairs if length > length_threshold]
                        filtered_count = len(filtered)
                        ignored_count = total_pairs - filtered_count
                        print(
                            "DEBUG finalize_block: try length>%s -> kept=%s ignored=%s min_required>%s"
                            % (
                                length_threshold,
                                filtered_count,
                                ignored_count,
                                WORD_LENGTH_FILTER_MIN_WORDS,
                            )
                        )
                        if filtered_count > WORD_LENGTH_FILTER_MIN_WORDS:
                            heights_for_ratio = filtered
                            applied_length_filter = length_threshold
                            ignored_tokens = [token for _, length, token in pairs if length <= length_threshold]
                            print(
                                "DEBUG finalize_block: applying length filter>%s; ignored_tokens=%s"
                                % (length_threshold, ignored_tokens)
                            )
                            break
                    if applied_length_filter is None:
                        print("DEBUG finalize_block: no length filter applied; using all words")

                count_above_h1 = sum(1 for h in heights_for_ratio if h >= average_height * HEADING_H1_RATIO)
                count_above_h2 = sum(1 for h in heights_for_ratio if h >= average_height * HEADING_H2_RATIO)
                total_words = len(heights_for_ratio)

                if total_words > 0:
                    # If this block was created by a negative horizontal split (back-split), require
                    # a minimum number of words to consider promoting it to a heading. This avoids
                    # single-word fragments (often caused by layout splits) being misclassified as h2/h1.
                    if block_data.get('split_reason') == 'horizontal_indent' and total_words < 3:
                        tag = 'p'
                        print(
                            f"DEBUG finalize_block: prevented promoting to h1/h2 due to negative-split and total_words={total_words} < 3"
                        )
                    # Additional guard: if the fragment starts with any quote-like character,
                    # it's very likely a continuation/dialogue line (OCR often puts opening
                    # quotes on a wrapped line). Such lines should not be promoted to headings.
                    elif block_data.get('split_reason') == 'horizontal_indent':
                        # Check leading character(s) of the textual content (strip leading whitespace)
                        leading = text_content.lstrip()[:1]
                        # Include a set of common opening quote characters (straight/directional)
                        quote_chars = set('"' + "'“”‹›«»‚‛‟`´ʺʹ")
                        if leading and leading in quote_chars:
                            tag = 'p'
                            print(f"DEBUG finalize_block: prevented promoting to h1/h2 because fragment starts with quote '{leading}' and split_reason=horizontal_indent")
                        else:
                            # fall through to normal heading checks below
                            pass
                    else:
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
                print(
                    "DEBUG finalize_block: text='%s...', tag=%s, max_font=%s, average_height=%s, word_heights=%s, "
                    "count_above_h1=%s, count_above_h2=%s, total_words=%s, min_ratio=%.3f, length_filter=%s"
                    % (
                        text_content[:100],
                        tag,
                        max_font,
                        average_height,
                        heights_for_ratio,
                        count_above_h1,
                        count_above_h2,
                        total_words,
                        min_ratio,
                        applied_length_filter,
                    )
                )

            blocks.append({
                'text': text_content,
                'tag': tag,
                'centered': block_data.get('centered', False),
                'all_bold': block_all_bold,
                'source_block_id': source_block_id,
                'line_font_sizes': line_font_sizes,
                'line_bold_flags': line_bold_flags,
                'line_count': len(block_data.get('lines', [])),
                'word_heights': list(effective_word_heights),
                'word_lengths': list(effective_word_lengths),
                'word_tokens': list(effective_word_tokens),
                'line_centers': list(block_data.get('line_centers', [])),
                'line_widths': list(block_data.get('line_widths', [])),
                'line_vpos': list(block_data.get('line_vpos', [])),
                'line_bottoms': list(block_data.get('line_bottoms', [])),
                'split_reason': block_data.get('split_reason'),
            })

        def representative_font_size(font_sizes: List[int]) -> Optional[float]:
            values = [size for size in font_sizes if size]
            if not values:
                return None
            return statistics.median(values)

        def should_merge_heading_blocks(first_block: Dict[str, Any], second_block: Dict[str, Any]) -> bool:
            if first_block.get('tag') not in {'h1', 'h2'}:
                return False
            if second_block.get('tag') != first_block.get('tag'):
                return False

            # Do not merge if either block was split due to gap or shift
            if first_block.get('split_reason') or second_block.get('split_reason'):
                print(f"DEBUG should_merge: not merging due to split_reason: first={first_block.get('split_reason')}, second={second_block.get('split_reason')}")
                return False

            first_source = first_block.get('source_block_id')
            second_source = second_block.get('source_block_id')
            if not first_source or not second_source or first_source != second_source:
                return False

            first_bold = first_block.get('all_bold', False)
            second_bold = second_block.get('all_bold', False)
            if first_bold != second_bold:
                return False

            first_font = representative_font_size(first_block.get('line_font_sizes', []))
            second_font = representative_font_size(second_block.get('line_font_sizes', []))
            if first_font is None or second_font is None:
                return False

            larger = max(first_font, second_font)
            smaller = min(first_font, second_font)
            if smaller <= 0:
                return False

            relative_diff = (larger - smaller) / smaller
            if relative_diff > HEADING_FONT_MERGE_TOLERANCE:
                return False

            print(f"DEBUG should_merge: merging blocks")
            return True

        def merge_heading_pair(first_block: Dict[str, Any], second_block: Dict[str, Any]) -> Dict[str, Any]:
            first_text = (first_block.get('text') or '').rstrip()
            second_text = (second_block.get('text') or '').lstrip()

            if first_text.endswith('-'):
                combined_text = (first_text[:-1] + second_text).strip()
            else:
                combined_text = (first_text + ' ' + second_text).strip()

            merged_block = dict(first_block)
            merged_block['text'] = re.sub(r'\s+', ' ', combined_text)
            merged_block['line_font_sizes'] = list(first_block.get('line_font_sizes', [])) + list(second_block.get('line_font_sizes', []))
            merged_block['line_bold_flags'] = list(first_block.get('line_bold_flags', [])) + list(second_block.get('line_bold_flags', []))
            merged_block['line_count'] = first_block.get('line_count', 0) + second_block.get('line_count', 0)
            merged_block['line_centers'] = list(first_block.get('line_centers', [])) + list(second_block.get('line_centers', []))
            merged_block['line_widths'] = list(first_block.get('line_widths', [])) + list(second_block.get('line_widths', []))
            merged_block['line_vpos'] = list(first_block.get('line_vpos', [])) + list(second_block.get('line_vpos', []))
            merged_block['line_bottoms'] = list(first_block.get('line_bottoms', [])) + list(second_block.get('line_bottoms', []))
            merged_block['centered'] = first_block.get('centered') or second_block.get('centered')
            merged_block['all_bold'] = first_block.get('all_bold', False) and second_block.get('all_bold', False)

            return merged_block

        def merge_heading_sequences(block_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            merged_blocks: List[Dict[str, Any]] = []
            index = 0

            while index < len(block_list):
                current_block_entry = block_list[index]
                if current_block_entry.get('tag') in {'h1', 'h2'}:
                    working_block = current_block_entry
                    cursor = index
                    while cursor + 1 < len(block_list) and should_merge_heading_blocks(working_block, block_list[cursor + 1]):
                        working_block = merge_heading_pair(working_block, block_list[cursor + 1])
                        cursor += 1
                    merged_blocks.append(working_block)
                    index = cursor + 1
                    continue

                merged_blocks.append(current_block_entry)
                index += 1

            return merged_blocks

        def adjust_single_line_centering(block_list: List[Dict[str, Any]]) -> None:
            def block_vertical_info(block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], float]:
                vpos = block.get('line_vpos', [])
                bottoms = block.get('line_bottoms', [])
                top_value = float(vpos[0]) if vpos else None
                bottom_value = float(bottoms[0]) if bottoms else None
                if top_value is not None and bottom_value is not None:
                    height = max(bottom_value - top_value, 1.0)
                else:
                    height = 1.0
                return top_value, bottom_value, height

            def vertical_pair_ok(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
                left_top, left_bottom, left_height = block_vertical_info(left)
                right_top, right_bottom, right_height = block_vertical_info(right)
                if left_bottom is None or right_top is None:
                    return True
                gap = right_top - left_bottom
                if gap <= 0:
                    return True
                reference_height = max(left_height, right_height, 1.0)
                return gap <= reference_height * SINGLE_LINE_VERTICAL_GAP_RATIO

            i = 0
            total = len(block_list)
            while i < total:
                if block_list[i].get('line_count') != 1:
                    i += 1
                    continue

                j = i
                while j < total and block_list[j].get('line_count') == 1:
                    if j > i and not vertical_pair_ok(block_list[j - 1], block_list[j]):
                        break
                    j += 1

                if j - i >= 2:
                    centers: List[float] = []
                    widths: List[float] = []
                    for block in block_list[i:j]:
                        centers.extend(
                            [
                                float(center)
                                for center in block.get('line_centers', [])
                                if center is not None
                            ]
                        )
                        widths.extend(
                            [
                                float(width)
                                for width in block.get('line_widths', [])
                                if width
                            ]
                        )

                    if len(centers) >= 2 and len(widths) >= 2:
                        median_center = statistics.median(centers)
                        median_width = statistics.median(widths)
                        margin = median_width * CENTER_ALIGNMENT_ERROR_MARGIN if median_width else 0
                        centers_aligned = all(
                            abs(center - median_center) <= margin for center in centers
                        )

                        min_width = min(widths)
                        max_width = max(widths)
                        width_diff = max_width - min_width
                        width_threshold = (
                            median_width * CENTER_ALIGNMENT_MIN_LINE_LEN_DIFF
                            if median_width
                            else 0
                        )
                        widths_vary = width_diff > width_threshold

                        if centers_aligned and widths_vary:
                            for block in block_list[i:j]:
                                block['centered'] = True

                i = j

        page_summary = (context or {}).get('page') or {}
        page_number_original = page_summary.get('pageNumber')
        page_number_value, page_number_is_searchable = self._normalize_page_number(page_number_original)
        page_number_annotations: List[Tuple[int, str]] = []
        page_number_line_ids: set[int] = set()

        # Zpracování TextBlocků
        text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextBlock')
        if not text_blocks:
            text_blocks = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextBlock')

        print(f"DEBUG: Found {len(text_blocks)} TextBlocks")

        if page_number_value or page_number_original:
            primary_annotation_added = False
            secondary_entries: List[Dict[str, Any]] = []

            if page_number_value and page_number_is_searchable:
                text_line_elements = root.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextLine')
                if not text_line_elements:
                    text_line_elements = root.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextLine')

                line_entries: List[Dict[str, Any]] = []
                width_values: List[int] = []

                for text_line in text_line_elements:
                    try:
                        line_width = int(text_line.get('WIDTH', '0') or 0)
                    except (TypeError, ValueError):
                        line_width = 0
                    try:
                        line_vpos = int(text_line.get('VPOS', '0') or 0)
                    except (TypeError, ValueError):
                        line_vpos = 0

                    if line_width > 0:
                        width_values.append(line_width)

                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                    if not strings:
                        strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')

                    line_parts: List[str] = []
                    alnum_segments: List[Tuple[int, int]] = []
                    for string_el in strings:
                        content = string_el.get('CONTENT', '') or ''
                        subs_content = string_el.get('SUBS_CONTENT', '') or ''
                        subs_type = string_el.get('SUBS_TYPE', '') or ''

                        content = html.unescape(content)
                        subs_content = html.unescape(subs_content)

                        try:
                            string_hpos = int(string_el.get('HPOS', '0') or 0)
                        except (TypeError, ValueError):
                            string_hpos = None
                        try:
                            string_width = int(string_el.get('WIDTH', '0') or 0)
                        except (TypeError, ValueError):
                            string_width = 0

                        if subs_type == 'HypPart1':
                            content_value = subs_content
                        elif subs_type == 'HypPart2':
                            continue
                        else:
                            content_value = content

                        if content_value:
                            line_parts.append(content_value)
                            if string_hpos is not None and any(ch.isalnum() for ch in content_value):
                                right_edge = string_hpos + max(string_width, 0)
                                alnum_segments.append((string_hpos, right_edge))

                    effective_width = line_width
                    if alnum_segments:
                        left_edge = min(segment[0] for segment in alnum_segments)
                        right_edge = max(segment[1] for segment in alnum_segments)
                        if right_edge > left_edge:
                            effective_width = right_edge - left_edge
                        else:
                            effective_width = 0

                    line_entries.append({
                        'element': text_line,
                        'width': line_width,
                        'effective_width': effective_width,
                        'vpos': line_vpos,
                        'text': ' '.join(line_parts).strip(),
                    })

                reference_width = alto_width2 or alto_width
                if reference_width and reference_width > 0:
                    short_threshold = max(reference_width * PAGE_NUMBER_SHORT_LINE_RATIO_TO_PAGE, 1)
                else:
                    short_threshold = None

                print(
                    f"DEBUG page-number: total_lines={len(line_entries)}, reference_width={reference_width}, short_threshold={short_threshold}"
                )

                target_has_letters = any(ch.isalpha() for ch in page_number_value) if page_number_value else False

                if short_threshold is not None:
                    if target_has_letters:
                        pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(page_number_value)}(?![0-9A-Za-z])")
                    else:
                        pattern = re.compile(rf"(?<!\d){re.escape(page_number_value)}(?!\d)")
                    sorted_entries = sorted(line_entries, key=lambda entry: entry['vpos'])

                    matches: List[Dict[str, Any]] = []
                    suspects: List[Dict[str, Any]] = []
                    number_found = False

                    for entry in sorted_entries:
                        width = entry.get('effective_width', entry['width'])
                        if width <= 0 or width > short_threshold:
                            print(
                                f"DEBUG page-number: skipping line width={width} vpos={entry['vpos']} text='{entry['text']}'"
                            )
                            continue

                        text_value = entry['text']
                        if not text_value:
                            print(
                                f"DEBUG page-number: skipping empty text for width={width} vpos={entry['vpos']}"
                            )
                            continue

                        nonspace_chars = [ch for ch in text_value if not ch.isspace()]
                        if len(nonspace_chars) >= PAGE_NUMBER_MIN_NONSPACE_FOR_REJECTION and not target_has_letters:
                            alpha_count = sum(1 for ch in nonspace_chars if ch.isalpha())
                            total_chars = len(nonspace_chars)
                            if alpha_count >= PAGE_NUMBER_ALPHA_REJECTION_RATIO * total_chars:
                                print(
                                    "DEBUG page-number: rejecting line due to alpha ratio "
                                    f"alpha={alpha_count} total={total_chars} width={width} "
                                    f"vpos={entry['vpos']} text='{text_value}'"
                                )
                                continue

                        if pattern.search(text_value):
                            print(
                                f"DEBUG page-number: FOUND match width={width} vpos={entry['vpos']} text='{text_value}'"
                            )
                            matches.append(entry)
                            number_found = True
                            suspects.clear()
                            continue

                        secondary_match = re.search(r"(?<!\d)(\d+)[\s\W]*\*", text_value)
                        if secondary_match:
                            match_start, match_end = secondary_match.span()
                            surrounding_text = text_value[:match_start] + text_value[match_end:]
                            has_other_alnum = bool(re.search(r"[0-9A-Za-z]", surrounding_text))
                            is_pure_secondary = not has_other_alnum
                            if is_pure_secondary:
                                print(
                                    f"DEBUG page-number: secondary detected (pure) vpos={entry['vpos']} text='{text_value}'"
                                )
                                page_number_line_ids.add(id(entry['element']))
                            else:
                                print(
                                    f"DEBUG page-number: secondary candidate vpos={entry['vpos']} text='{text_value}'"
                                )
                            secondary_entries.append({
                                'text': text_value,
                                'is_pure': is_pure_secondary,
                            })
                            continue

                        stripped_value = text_value.strip()
                        if re.fullmatch(r"\d{1,4}", stripped_value):
                            print(
                                "DEBUG page-number: secondary numeric candidate "
                                f"vpos={entry['vpos']} text='{text_value}'"
                            )
                            # Mark simple numeric lines as pure secondary candidates and
                            # keep the original entry so we can promote it to primary later
                            secondary_entries.append({
                                'text': text_value,
                                'is_pure': True,
                                'entry': entry,
                            })
                            continue

                        if not number_found:
                            print(
                                f"DEBUG page-number: candidate width={width} vpos={entry['vpos']} text='{text_value}'"
                            )
                            suspects.append(entry)

                    candidates = matches if matches else suspects
                    # If we found no matches/suspects but have exactly one pure numeric
                    # secondary candidate, promote it to primary. This covers OCR errors
                    # where the OCR reads a different number than metadata but a clear
                    # numeric line exists on the page.
                    if not candidates and not matches and not suspects and len(secondary_entries) == 1:
                        sec = secondary_entries[0]
                        if sec.get('is_pure') and sec.get('entry'):
                            promoted_entry = sec['entry']
                            print(
                                f"DEBUG page-number: promoting single pure secondary '{sec.get('text')}' to primary candidate vpos={promoted_entry['vpos']}"
                            )
                            candidates = [promoted_entry]
                            # Remove the promoted secondary so it won't be reported again
                            secondary_entries.clear()
                            # Ensure the promoted line is treated as primary (skip it in later text processing)
                            try:
                                page_number_line_ids.add(id(promoted_entry['element']))
                            except Exception:
                                pass
                    print(
                        f"DEBUG page-number: matches={len(matches)}, suspects={len(suspects)}, using_candidates={len(candidates)}"
                    )

                    if candidates:
                        annotation_is_found = bool(matches)
                        for entry in candidates:
                            if annotation_is_found:
                                page_number_line_ids.add(id(entry['element']))
                            ocr_text_display = entry['text'] or ''
                            annotation_parts = [
                                f"Page number: {page_number_value}",
                                f"OCR text: {ocr_text_display}",
                            ]
                            if not annotation_is_found:
                                annotation_parts.append("!FOUND ONLY KANDIDATE!")
                            annotation_text = ' ; '.join(annotation_parts)
                            page_number_annotations.append(
                                (
                                    entry['vpos'],
                                    f"<note{PAGE_NOTE_STYLE_ATTR}>{html.escape(annotation_text, quote=False)}</note>"
                                )
                            )
                            primary_annotation_added = True
                    else:
                        print("DEBUG page-number: no candidates detected after scanning")

                else:
                    print("DEBUG page-number: cannot compute short_threshold due to missing page width")

                if not primary_annotation_added:
                    reference_height_local = alto_height2 or alto_height
                    fallback_text = f"Page number: {page_number_value} ; OCR text: None ; !NOT FOUND!"
                    fallback_vpos = (reference_height_local or 0) + 1
                    page_number_annotations.append(
                        (
                            fallback_vpos,
                            f"<note{PAGE_NOTE_STYLE_ATTR}>{html.escape(fallback_text, quote=False)}</note>"
                        )
                    )
                    primary_annotation_added = True
                    print(
                        "DEBUG page-number: appended fallback annotation due to missing candidates"
                    )

            else:
                reference_height_local = alto_height2 or alto_height
                display_value = ''
                if page_number_original is not None:
                    display_value = str(page_number_original).strip()
                if not display_value and page_number_value:
                    display_value = page_number_value

                if display_value:
                    fallback_vpos = (reference_height_local or 0) + 1
                    annotation_text = f"Page number: {display_value} ; SHOULDN'T BE ON PAGE"
                    page_number_annotations.append(
                        (
                            fallback_vpos,
                            f"<note{PAGE_NOTE_STYLE_ATTR}>{html.escape(annotation_text, quote=False)}</note>"
                        )
                    )
                    print("DEBUG page-number: skipped detection for non-numeric metadata")
                primary_annotation_added = True

            if secondary_entries:
                reference_height_local = alto_height2 or alto_height
                current_max_vpos = max(
                    (existing_vpos for existing_vpos, _ in page_number_annotations),
                    default=(reference_height_local or 0)
                )
                base_vpos = max(current_max_vpos, (reference_height_local or 0)) + 1
                for index, secondary in enumerate(secondary_entries):
                    suffix = '' if secondary['is_pure'] else ' ; !FOUND OLNY KANDIDATE!'
                    annotation_text = (
                        f"Secondary page number OCR: {secondary['text']}{suffix}"
                    )
                    page_number_annotations.append(
                        (
                            base_vpos + index,
                            f"<note{PAGE_NOTE_STYLE_ATTR}>{html.escape(annotation_text, quote=False)}</note>"
                        )
                    )

        # Pre-scan all text lines to build font_counts based on characters
        font_counts = defaultdict(int)
        for block_elem in text_blocks:
            text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}TextLine')
            if not text_lines:
                text_lines = block_elem.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}TextLine')
            for text_line in text_lines:
                if page_number_line_ids and id(text_line) in page_number_line_ids:
                    continue

                text_line_width = int(text_line.get('WIDTH', '0') or 0)
                if text_line_width <= 0:
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
                if page_number_line_ids and id(text_line) in page_number_line_ids:
                    continue

                text_line_width = int(text_line.get('WIDTH', '0') or 0)
                if text_line_width <= 0:
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

            current_block = {
                'lines': [],
                'tag': 'p',
                'font_sizes': set(),
                'word_heights': [],
                'word_lengths': [],
                'word_tokens': [],
                'line_word_heights': [],
                'line_word_lengths': [],
                'line_word_tokens': [],
                'all_bold': True,
            }

            line_heights = [record['height'] for record in line_records]
            line_widths = [record['width'] for record in line_records]

            vertical_gaps = []
            horizontal_shifts = []
            previous_bottom = None
            previous_left = None

            # Calculate line centers for center alignment detection
            line_centers = [record['hpos'] + record['width'] / 2 for record in line_records]
            median_center = statistics.median(line_centers) if line_centers else 0
            page_center = alto_width / 2
            median_width = statistics.median(line_widths) if line_widths else 0
            margin_median = median_width * CENTER_ALIGNMENT_ERROR_MARGIN if median_width else 0
            margin_center = alto_width * CENTER_ALIGNMENT_ERROR_MARGIN if alto_width else 0

            # Check if line widths vary enough (not a justified block) overall
            if line_widths:
                min_width = min(line_widths)
                max_width = max(line_widths)
                width_diff = max_width - min_width
                width_diff_threshold = median_width * CENTER_ALIGNMENT_MIN_LINE_LEN_DIFF if median_width else 0
                widths_vary_overall = width_diff > width_diff_threshold
            else:
                widths_vary_overall = False

            # We'll compute vertical/horizontal gaps and thresholds first; subgrouping will follow after thresholds are set
            group_centered_flags: List[bool] = []
            record_group_idxs: List[int] = []

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

            # overall center decision (used for debugging fallback)
            is_center_aligned = (all(abs(center - median_center) <= margin_median for center in line_centers) or all(abs(center - page_center) <= margin_center for center in line_centers)) and widths_vary_overall
            print(f"DEBUG: line_heights={line_heights}")
            print(f"DEBUG: line_widths={line_widths}")
            print(f"DEBUG: vertical_gaps={vertical_gaps}, horizontal_shifts={horizontal_shifts}")
            print(f"DEBUG: median_height={median_height}, median_width={median_width}, median_gap={median_gap}, median_shift={median_shift}")
            print(f"DEBUG: trimmed_shifts={trimmed_shifts}")
            print(f"DEBUG: vertical_threshold_candidates={vertical_threshold_candidates}, horizontal_threshold_candidates={horizontal_threshold_candidates}")
            print(f"DEBUG: vertical_threshold={vertical_threshold}, horizontal_threshold={horizontal_threshold}")
            print(f"DEBUG: heading_fonts={heading_fonts}, font_counts={dict(font_counts)}")
            print(f"DEBUG: is_center_aligned={is_center_aligned}")

            # Now split the TextBlock into vertical subgroups using the computed vertical_threshold
            groups: List[List[dict]] = []
            current_group: List[dict] = []
            prev_bottom_local = None
            for rec in line_records:
                if prev_bottom_local is not None:
                    gap_local = rec['vpos'] - prev_bottom_local
                    if gap_local > vertical_threshold:
                        if current_group:
                            groups.append(current_group)
                        current_group = []
                current_group.append(rec)
                prev_bottom_local = rec['bottom']
            if current_group:
                groups.append(current_group)

            # Compute center-alignment flag for each subgroup (centers aligned AND widths vary within subgroup)
            for gidx, grp in enumerate(groups):
                centers_grp = [r['hpos'] + r['width'] / 2 for r in grp]
                median_center_grp = statistics.median(centers_grp) if centers_grp else 0
                median_width_grp = statistics.median([r['width'] for r in grp]) if grp else 0
                margin_median_grp = median_width_grp * CENTER_ALIGNMENT_ERROR_MARGIN if median_width_grp else 0

                centers_aligned_median_grp = all(abs(c - median_center_grp) <= margin_median_grp for c in centers_grp) if centers_grp else False
                centers_aligned_page_grp = all(abs(c - page_center) <= margin_center for c in centers_grp) if centers_grp else False
                centers_aligned_grp = centers_aligned_median_grp or centers_aligned_page_grp

                widths_vary_grp = False
                if grp:
                    widths = [r['width'] for r in grp]
                    min_w = min(widths)
                    max_w = max(widths)
                    width_diff_grp = max_w - min_w
                    width_diff_threshold_grp = median_width_grp * CENTER_ALIGNMENT_MIN_LINE_LEN_DIFF if median_width_grp else 0
                    widths_vary_grp = width_diff_grp > width_diff_threshold_grp

                group_centered = centers_aligned_grp and widths_vary_grp
                group_centered_flags.append(group_centered)

                # record mapping for each line in group
                for _ in grp:
                    record_group_idxs.append(gidx)

            # Extra debug: report group info for specific lines of interest
            interesting_a = 'O mluvícím ptáku, živé vodě a třech'
            interesting_b = 'zlatých jabloních.'
            for ridx, rec in enumerate(line_records):
                # try to extract a text snippet from the underlying element (may be filled later)
                try:
                    el = rec.get('element')
                    strings = el.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                    if not strings:
                        strings = el.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')
                    snippet = ' '.join([html.unescape(s.get('CONTENT','') or s.get('SUBS_CONTENT','') or '') for s in strings]).strip()
                except Exception:
                    snippet = ''

                if interesting_a in snippet or interesting_b in snippet:
                    gidx = record_group_idxs[ridx] if ridx < len(record_group_idxs) else None
                    group_cent = group_centered_flags[gidx] if (gidx is not None and gidx < len(group_centered_flags)) else None
                    centers_grp = [r['hpos'] + r['width'] / 2 for r in groups[gidx]] if gidx is not None and gidx < len(groups) else []
                    median_center_grp = statistics.median(centers_grp) if centers_grp else None
                    median_width_grp = statistics.median([r['width'] for r in groups[gidx]]) if gidx is not None and gidx < len(groups) and groups[gidx] else None
                    print(
                        f"DEBUG-INTEREST idx={ridx} text='{snippet[:80]}...' group={gidx} group_centered={group_cent} median_center_grp={median_center_grp} median_width_grp={median_width_grp} vertical_threshold={vertical_threshold} horizontal_threshold={horizontal_threshold}"
                    )

            # If no groups found, fallback to overall decision
            if not group_centered_flags:
                is_center_aligned = (all(abs(center - median_center) <= margin_median for center in line_centers) or all(abs(center - page_center) <= margin_center for center in line_centers)) and widths_vary_overall
                group_centered_flags = [is_center_aligned]
                record_group_idxs = [0 for _ in line_records]

            tag = 'p'
            block_id = block_elem.get('ID')

            def new_block_state(centered: bool = False):
                return {
                    'lines': [],
                    'tag': tag,
                    'font_sizes': set(),
                    'word_heights': [],
                    'word_lengths': [],
                    'word_tokens': [],
                    'line_word_heights': [],
                    'line_word_lengths': [],
                    'line_word_tokens': [],
                    'centered': centered,
                    'all_bold': True,
                    'source_block_id': block_id,
                    'line_font_sizes': [],
                    'line_bold_flags': [],
                    'line_centers': [],
                    'line_widths': [],
                    'line_vpos': [],
                    'line_bottoms': [],
                    'split_reason': None,
                }

            # start with centered flag of the first subgroup
            first_centered = group_centered_flags[record_group_idxs[0]] if record_group_idxs else False
            current_block = new_block_state(centered=first_centered)
            lines = 0
            last_bottom = None
            last_left = None
            last_line_font_size = 0

            for idx, record in enumerate(line_records):
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
                            current_block['split_reason'] = 'vertical_gap'
                            finalize_block(
                                current_block,
                                current_block.get('word_heights', []),
                                current_block.get('word_lengths', []),
                                current_block.get('word_tokens', []),
                                average_height,
                                heading_fonts,
                            )
                        # Start a new block and inherit the centered flag for the subgroup this line belongs to
                        grp_idx_for_record = record_group_idxs[idx] if idx < len(record_group_idxs) else None
                        next_centered = False
                        if grp_idx_for_record is not None and grp_idx_for_record < len(group_centered_flags):
                            next_centered = group_centered_flags[grp_idx_for_record]
                        current_block = new_block_state(centered=next_centered)
                        lines = 0

                last_bottom = bottom

                strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v3#}String')
                if not strings:
                    strings = text_line.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String')

                line_parts: List[str] = []
                line_all_bold = True
                current_line_word_heights: List[float] = []
                current_line_word_lengths: List[int] = []
                current_line_word_tokens: List[str] = []
                # Token-level arrays used for effective-left calculation
                current_line_token_hpos: List[Optional[int]] = []
                current_line_token_texts: List[str] = []

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

                    # When a word is hyphen-split across lines, ALTO uses SUBS_CONTENT with
                    # SUBS_TYPE 'HypPart1'/'HypPart2'. To compute effective_left correctly we
                    # must consider the HypPart2 token's HPOS on the continuation line. However
                    # we should avoid duplicating the full SUBS_CONTENT into the visible line
                    # text (that was already included on the previous line). Therefore:
                    # - For HypPart1: use SUBS_CONTENT as the visible content_value (same as before)
                    # - For HypPart2: do NOT append to line_parts (leave content_value empty) but
                    #   include the SUBS_CONTENT in the token-level arrays so its HPOS is considered
                    #   when computing effective_left.
                    if subs_type == 'HypPart1':
                        content_value = subs_content
                        token_text_for_hpos = content_value
                    elif subs_type == 'HypPart2':
                        # continuation fragment: use the full reconstructed word for HPOS purposes,
                        # but don't add it to line_parts or word metrics to avoid duplication.
                        content_value = ''
                        token_text_for_hpos = subs_content or content
                    else:
                        content_value = content
                        token_text_for_hpos = content_value

                    token_text = (token_text_for_hpos or '')
                    # token-level HPOS (fallback to line hpos if missing)
                    try:
                        token_hpos = int(string_el.get('HPOS')) if string_el.get('HPOS') else None
                    except (TypeError, ValueError):
                        token_hpos = None
                    # Always record token HPOS/text for effective-left computation, including HypPart2
                    current_line_token_texts.append(token_text)
                    current_line_token_hpos.append(token_hpos)
                    non_space_length = len([ch for ch in token_text if not ch.isspace()])

                    height = string_el.get('HEIGHT')
                    if height:
                        try:
                            height_value = float(height)
                        except (TypeError, ValueError):
                            height_value = None
                        if height_value is not None and content_value:
                            current_line_word_heights.append(height_value)
                            current_line_word_lengths.append(non_space_length)
                            current_line_word_tokens.append(token_text)

                    if content_value:
                        line_parts.append(content_value)

                line_text = ' '.join(line_parts).strip()
                line_width = record['width']
                line_center = text_line_hpos + line_width / 2 if line_width else float(text_line_hpos)
                # compute effective left using first non-noise token HPOS if available
                def compute_effective_left(token_texts: List[str], token_hpos: List[Optional[int]], fallback: int) -> int:
                    for i in range(min(len(token_texts), EFFECTIVE_LEFT_MAX_TOKEN_SCAN)):
                        t = (token_texts[i] or '').strip()
                        if not t:
                            continue
                        # if token contains alnum, consider it significant
                        if any(ch.isalnum() for ch in t):
                            return token_hpos[i] if token_hpos[i] is not None else fallback
                        # if token is single punctuation likely OCR noise, skip it
                        if len(t) == 1 and t in NOISE_LEADING_PUNCT:
                            continue
                        # otherwise treat as significant
                        return token_hpos[i] if token_hpos[i] is not None else fallback
                    return fallback

                text_line_effective_left = compute_effective_left(current_line_token_texts, current_line_token_hpos, text_line_hpos)
                previous_left = last_left
                print(
                    f"DEBUG: Processing line at hpos={text_line_hpos} (effective_left={text_line_effective_left}), previous_left={previous_left}, line_text='{line_text[:50]}...'"
                )

                if not line_text:
                    last_left = text_line_hpos
                    last_line_font_size = current_line_font_size
                    continue

                filtered_current_heights = [h for h in current_line_word_heights if h > 0]
                current_line_avg_height = (
                    statistics.mean(filtered_current_heights) if filtered_current_heights else None
                )

                previous_line_avg_height = None
                if current_block['line_word_heights']:
                    previous_line_heights = [
                        h for h in current_block['line_word_heights'][-1] if h and h > 0
                    ]
                    if previous_line_heights:
                        previous_line_avg_height = statistics.mean(previous_line_heights)

                height_ratio = None
                if previous_line_avg_height and current_line_avg_height:
                    smaller = min(previous_line_avg_height, current_line_avg_height)
                    if smaller > 0:
                        larger = max(previous_line_avg_height, current_line_avg_height)
                        height_ratio = larger / smaller

                font_size_ratio = None
                font_size_differs = False
                if last_line_font_size and current_line_font_size:
                    font_size_ratio = max(last_line_font_size, current_line_font_size) / min(
                        last_line_font_size, current_line_font_size
                    )
                    if font_size_ratio >= FONT_SIZE_SPLIT_RATIO:
                        font_size_differs = True

                should_split_center = False
                if is_center_aligned and current_block['lines']:
                    height_differs = height_ratio is not None and height_ratio >= CENTER_LINE_HEIGHT_RATIO
                    if font_size_differs or height_differs:
                        should_split_center = True

                # Only consider horizontal shifts for non-centered current blocks.
                # If the current block/subgroup is centered, horizontal shift is not informative and should be ignored.
                if previous_left is not None and not current_block.get('centered', False):
                    # use effective left positions for h_diff calculation
                    left_for_prev = previous_left
                    if isinstance(previous_left, (int, float)):
                        # previous_left is a simple number (line hpos); try to use stored effective if available
                        left_for_prev = previous_left
                    h_diff = text_line_effective_left - left_for_prev
                    print(
                        f"DEBUG: h_diff={h_diff}, horizontal_threshold={horizontal_threshold}, font_size_differs={font_size_differs}, effective_prev_left={left_for_prev}, effective_cur_left={text_line_effective_left}"
                    )

                    if h_diff > horizontal_threshold or font_size_differs:
                        print(
                            f"DEBUG: Horizontal split on h_diff={h_diff} > {horizontal_threshold} or font_size_differs={font_size_differs}"
                        )
                        if current_block['lines']:
                            current_block['split_reason'] = 'horizontal_shift'
                            finalize_block(
                                current_block,
                                current_block.get('word_heights', []),
                                current_block.get('word_lengths', []),
                                current_block.get('word_tokens', []),
                                average_height,
                                heading_fonts,
                            )
                        # Create a new block and inherit the centered flag of the subgroup the current line belongs to
                        grp_idx_for_record = record_group_idxs[idx] if idx < len(record_group_idxs) else None
                        next_centered = current_block.get('centered', False)
                        if grp_idx_for_record is not None and grp_idx_for_record < len(group_centered_flags):
                            next_centered = group_centered_flags[grp_idx_for_record]
                        current_block = new_block_state(centered=next_centered)
                        lines = 0
                    elif h_diff < 0 and abs(h_diff) > horizontal_threshold * NEGATIVE_SHIFT_MULTIPLIER and current_block['lines']:
                        print(
                            f"DEBUG: Negative split on h_diff={h_diff}, abs(h_diff)={abs(h_diff)} > {horizontal_threshold * NEGATIVE_SHIFT_MULTIPLIER}"
                        )
                        if len(current_block['lines']) > 1:
                            for idx, previous_line in enumerate(current_block['lines'][:-1]):
                                line_font_size = (
                                    current_block['line_font_sizes'][idx]
                                    if idx < len(current_block['line_font_sizes'])
                                    else 0
                                )
                                line_bold_flag = (
                                    current_block['line_bold_flags'][idx]
                                    if idx < len(current_block['line_bold_flags'])
                                    else False
                                )
                                previous_line_word_heights = (
                                    current_block['line_word_heights'][idx]
                                    if idx < len(current_block['line_word_heights'])
                                    else []
                                )
                                previous_line_word_lengths = (
                                    current_block['line_word_lengths'][idx]
                                    if idx < len(current_block['line_word_lengths'])
                                    else []
                                )
                                previous_line_word_tokens = (
                                    current_block['line_word_tokens'][idx]
                                    if idx < len(current_block['line_word_tokens'])
                                    else []
                                )
                                previous_line_center = (
                                    current_block['line_centers'][idx]
                                    if idx < len(current_block['line_centers'])
                                    else None
                                )
                                previous_line_width = (
                                    current_block['line_widths'][idx]
                                    if idx < len(current_block['line_widths'])
                                    else None
                                )
                                previous_line_vpos = (
                                    current_block['line_vpos'][idx]
                                    if idx < len(current_block['line_vpos'])
                                    else None
                                )
                                previous_line_bottom = (
                                    current_block['line_bottoms'][idx]
                                    if idx < len(current_block['line_bottoms'])
                                    else None
                                )
                                finalize_block(
                                    {
                                        'lines': [previous_line],
                                        'tag': tag,
                                        'font_sizes': {line_font_size} if line_font_size else set(),
                                        'word_heights': list(previous_line_word_heights),
                                        'word_lengths': list(previous_line_word_lengths),
                                        'word_tokens': list(previous_line_word_tokens),
                                        'centered': is_center_aligned,
                                        'all_bold': line_bold_flag,
                                        'source_block_id': block_id,
                                        'line_font_sizes': [line_font_size],
                                        'line_bold_flags': [line_bold_flag],
                                        'line_word_heights': [list(previous_line_word_heights)],
                                        'line_word_lengths': [list(previous_line_word_lengths)],
                                        'line_word_tokens': [list(previous_line_word_tokens)],
                                        'line_centers': [previous_line_center] if previous_line_center is not None else [],
                                        'line_widths': [previous_line_width] if previous_line_width is not None else [],
                                        'line_vpos': [previous_line_vpos] if previous_line_vpos is not None else [],
                                        'line_bottoms': [previous_line_bottom] if previous_line_bottom is not None else [],
                                        'split_reason': 'horizontal_indent',
                                    },
                                    list(previous_line_word_heights),
                                    list(previous_line_word_lengths),
                                    list(previous_line_word_tokens),
                                    average_height,
                                    heading_fonts,
                                )

                            last_line_text = current_block['lines'][-1]
                            last_font_size = (
                                current_block['line_font_sizes'][-1]
                                if current_block['line_font_sizes']
                                else 0
                            )
                            last_bold_flag = (
                                current_block['line_bold_flags'][-1]
                                if current_block['line_bold_flags']
                                else True
                            )
                            last_line_word_heights = (
                                current_block['line_word_heights'][-1]
                                if current_block['line_word_heights']
                                else []
                            )
                            last_line_word_lengths = (
                                current_block['line_word_lengths'][-1]
                                if current_block['line_word_lengths']
                                else []
                            )
                            last_line_word_tokens = (
                                current_block['line_word_tokens'][-1]
                                if current_block['line_word_tokens']
                                else []
                            )
                            last_line_center = (
                                current_block['line_centers'][-1]
                                if current_block['line_centers']
                                else None
                            )
                            last_line_width = (
                                current_block['line_widths'][-1]
                                if current_block['line_widths']
                                else None
                            )
                            last_line_vpos = (
                                current_block['line_vpos'][-1]
                                if current_block['line_vpos']
                                else None
                            )
                            last_line_bottom = (
                                current_block['line_bottoms'][-1]
                                if current_block['line_bottoms']
                                else None
                            )

                            current_block = new_block_state()
                            current_block['lines'].append(last_line_text)
                            current_block['line_font_sizes'].append(last_font_size)
                            current_block['line_bold_flags'].append(last_bold_flag)
                            if last_font_size:
                                current_block['font_sizes'].add(last_font_size)
                            if last_line_word_heights:
                                current_block['word_heights'].extend(last_line_word_heights)
                                current_block['word_lengths'].extend(last_line_word_lengths)
                                current_block['word_tokens'].extend(last_line_word_tokens)
                            current_block['line_word_heights'].append(list(last_line_word_heights))
                            current_block['line_word_lengths'].append(list(last_line_word_lengths))
                            current_block['line_word_tokens'].append(list(last_line_word_tokens))
                            if last_line_center is not None:
                                current_block['line_centers'].append(last_line_center)
                            if last_line_width is not None:
                                current_block['line_widths'].append(last_line_width)
                            if last_line_vpos is not None:
                                current_block['line_vpos'].append(last_line_vpos)
                            if last_line_bottom is not None:
                                current_block['line_bottoms'].append(last_line_bottom)
                            current_block['all_bold'] = last_bold_flag
                            lines = 1
                elif should_split_center:
                    print(
                        "DEBUG: Center-aligned block split due to font/height difference: "
                        f"font_size_ratio={font_size_ratio}, height_ratio={height_ratio}"
                    )
                    # remember whether current block was centered so the new block can inherit it
                    was_centered = bool(current_block.get('centered', False))
                    current_block['split_reason'] = 'center_font_change'
                    finalize_block(
                        current_block,
                        current_block.get('word_heights', []),
                        current_block.get('word_lengths', []),
                        current_block.get('word_tokens', []),
                        average_height,
                        heading_fonts,
                    )
                    # create a new block preserving the centered flag only if it was set
                    current_block = new_block_state(centered=was_centered)
                    lines = 0

                if current_line_font_size:
                    current_block['font_sizes'].add(current_line_font_size)

                if current_line_word_heights:
                    current_block['word_heights'].extend(current_line_word_heights)
                    current_block['word_lengths'].extend(current_line_word_lengths)
                    current_block['word_tokens'].extend(current_line_word_tokens)
                current_block['line_word_heights'].append(list(current_line_word_heights))
                current_block['line_word_lengths'].append(list(current_line_word_lengths))
                current_block['line_word_tokens'].append(list(current_line_word_tokens))

                current_block['line_centers'].append(line_center)
                current_block['line_widths'].append(line_width)
                current_block['line_vpos'].append(float(text_line_vpos))
                current_block['line_bottoms'].append(float(bottom))

                current_block['lines'].append(line_text)
                current_block['line_font_sizes'].append(current_line_font_size)
                current_block['line_bold_flags'].append(line_all_bold)

                if not line_all_bold:
                    current_block['all_bold'] = False

                prospective_count = lines + 1
                lines = len(current_block['lines'])

                # store effective left as last_left for next iteration
                last_left = text_line_effective_left
                last_line_font_size = current_line_font_size

            if current_block['lines']:
                finalize_block(
                    current_block,
                    current_block.get('word_heights', []),
                    current_block.get('word_lengths', []),
                    current_block.get('word_tokens', []),
                    average_height,
                    heading_fonts,
                )

        # Spojit víceliniové nadpisy podle dohodnutých heuristik
        blocks = merge_heading_sequences(blocks)
        adjust_single_line_centering(blocks)

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

        if average_height is not None:
            small_height_threshold = average_height * SMALL_RATIO
            for block in blocks:
                original_tag = block.get('tag')
                if original_tag not in ('p', 'h3'):
                    continue

                # Build triples from block tokens so we can apply same filters as in finalize_block
                word_heights = list(block.get('word_heights', []))
                word_lengths = list(block.get('word_lengths', []))
                word_tokens = list(block.get('word_tokens', []))

                pairs = list(zip(word_heights, word_lengths, word_tokens))
                # Keep only valid positive heights
                pairs = [(h, l, t) for (h, l, t) in pairs if h and h > 0]
                if not pairs:
                    continue

                # Remove punctuation-only tokens
                punct_removed = [t for _, _, t in pairs if not any(ch.isalnum() for ch in (t or ""))]
                if punct_removed:
                    print(f"DEBUG small: removing punctuation-only tokens before ratio: {punct_removed}")
                pairs = [p for p in pairs if any(ch.isalnum() for ch in (p[2] or ""))]
                if not pairs:
                    continue

                # Now apply the same length filter logic used for headings
                heights_for_small = [h for h, _, _ in pairs]
                applied_length_filter_small = None
                if pairs and WORD_LENGTH_FILTER_INITIAL > 0:
                    total_pairs = len(pairs)
                    print(
                        "DEBUG small: length-filter precheck total_pairs=%s initial_threshold=%s"
                        % (total_pairs, WORD_LENGTH_FILTER_INITIAL)
                    )
                    for length_threshold in range(WORD_LENGTH_FILTER_INITIAL, 0, -1):
                        filtered = [h for h, length, _ in pairs if length > length_threshold]
                        filtered_count = len(filtered)
                        ignored_count = total_pairs - filtered_count
                        print(
                            "DEBUG small: try length>%s -> kept=%s ignored=%s min_required>%s"
                            % (
                                length_threshold,
                                filtered_count,
                                ignored_count,
                                WORD_LENGTH_FILTER_MIN_WORDS,
                            )
                        )
                        if filtered_count > WORD_LENGTH_FILTER_MIN_WORDS:
                            heights_for_small = filtered
                            applied_length_filter_small = length_threshold
                            ignored_tokens = [token for _, length, token in pairs if length <= length_threshold]
                            print(
                                "DEBUG small: applying length filter>%s; ignored_tokens=%s"
                                % (length_threshold, ignored_tokens)
                            )
                            break

                valid_word_heights = list(heights_for_small)
                if not valid_word_heights:
                    continue

                small_word_count = sum(1 for h in valid_word_heights if h < small_height_threshold)
                total_valid = len(valid_word_heights)
                if total_valid == 0:
                    continue

                small_ratio_value = small_word_count / total_valid
                ratio_threshold = SMALL_MIN_WORD_RATIO
                # Ulevit prah pro malé bloky pokud byl původní tag 'h3'
                # nebo pokud odstavec začíná znakem '*'
                block_text_for_check = (block.get('text') or '').lstrip()
                if original_tag == 'h3' or block_text_for_check.startswith('*'):
                    ratio_threshold *= SMALL_RATIO_MULTIPLIER

                if small_ratio_value >= ratio_threshold:
                    block['tag'] = 'small'
                    print(
                        "DEBUG small demotion: demoted to <small>, "
                        f"source_tag={original_tag}, ratio={small_ratio_value:.3f}, "
                        f"required_ratio={ratio_threshold:.3f}, threshold={small_height_threshold:.2f}"
                    )

        # Generování HTML - přesně jako v TypeScript
        result = ""
        for block in blocks:
            text = block['text'].strip()
            if not text:  # Jen neprázdné bloky
                continue

            style_attr = ''
            if block.get('centered'):
                style_attr = ' style="text-align: center;"'

            if block['tag'] == 'small':
                result += f"<p{style_attr}><small>{text}</small></p>"
            else:
                result += f"<{block['tag']}{style_attr}>{text}</{block['tag']}>"

        if page_number_annotations:
            sorted_annotations = sorted(page_number_annotations, key=lambda item: item[0])
            annotation_fragments = [fragment for _, fragment in sorted_annotations]
            annotation_html = '<br>'.join(annotation_fragments)
            if result and annotation_html:
                result += '<br>' + annotation_html
            else:
                result += annotation_html

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
