from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import FAQEntry, load_csv_data

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    try:
        from PyPDF2 import PdfReader
    except ImportError:  # pragma: no cover - optional dependency
        PdfReader = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DATA_DIR = BASE_DIR / "data"
IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
DATA_DIR = (
    Path(os.getenv("HYBRID_AI_RUNTIME_DIR", str(Path(tempfile.gettempdir()) / "hybrid_ai")))
    if IS_SERVERLESS
    else PROJECT_DATA_DIR
)
FAQ_FILE = BASE_DIR / "faq_data.csv"
STORE_FILE = DATA_DIR / "knowledge_store.json"
BUNDLED_STORE_FILE = PROJECT_DATA_DIR / "knowledge_store.json"
UPLOADED_FILE_TYPES = {"csv", "json", "txt", "pdf", "docx"}
TERM_SYNONYMS = {
    "pay": {"pay", "payment", "payments", "paid", "payout", "salary", "wage", "billing", "invoice"},
    "payment": {"pay", "payment", "payments", "paid", "payout", "salary", "wage", "billing", "invoice"},
    "money": {"money", "payment", "payments", "paid", "salary", "wage", "amount", "cost", "price"},
    "refund": {"refund", "return", "reimbursement"},
    "delivery": {"delivery", "shipping", "shipment", "dispatch"},
    "cancel": {"cancel", "cancellation", "terminate", "stop"},
    "discount": {"discount", "offer", "promotion", "deal"},
}


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    source_name: str
    source_type: str
    text: str
    score: float = 0.0


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):  # noqa: ANN001
        if self._skip_depth:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self._parts.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self._parts)


class KnowledgeBase:
    def __init__(self, faq_file: Path = FAQ_FILE, store_file: Path = STORE_FILE):
        self.faq_file = faq_file
        self.store_file = store_file
        self.faq_entries: list[FAQEntry] = []
        self.extra_sources: list[dict] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._search_chunks: list[KnowledgeChunk] = []
        self._search_matrix = None
        self._normalized_search_texts: list[str] = []
        self._normalized_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(3, 5),
            analyzer="char_wb",
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._normalized_search_matrix = None
        self.reload()

    def reload(self) -> None:
        self._prepare_store_file()
        self.faq_entries = load_csv_data(str(self.faq_file))
        self.extra_sources = self._load_store()
        self._rebuild_index()

    def _prepare_store_file(self) -> None:
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        if IS_SERVERLESS and not self.store_file.exists() and BUNDLED_STORE_FILE.exists():
            shutil.copyfile(BUNDLED_STORE_FILE, self.store_file)

    def _load_store(self) -> list[dict]:
        if not self.store_file.exists():
            return []
        raw_bytes = self.store_file.read_bytes()
        payload = None

        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                payload = json.loads(raw_bytes.decode(encoding))
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        if payload is None:
            raise ValueError("knowledge_store.json could not be decoded as valid JSON.")

        return payload.get("sources", [])

    def _save_store(self) -> None:
        payload = {"sources": self.extra_sources}
        temp_file = self.store_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        temp_file.replace(self.store_file)

    def _rebuild_index(self) -> None:
        chunks: list[KnowledgeChunk] = []

        for faq in self.faq_entries:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=str(uuid.uuid4()),
                    source_id="faq-core",
                    source_name=faq.question,
                    source_type="faq",
                    text=f"Question: {faq.question}\nAnswer: {faq.answer}",
                )
            )

        for source in self.extra_sources:
            for chunk in source.get("chunks", []):
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk["chunk_id"],
                        source_id=source["source_id"],
                        source_name=source["source_name"],
                        source_type=source["source_type"],
                        text=chunk["text"],
                    )
                )

        self._search_chunks = chunks
        if chunks:
            raw_texts = [chunk.text for chunk in chunks]
            self._normalized_search_texts = [self._normalize_for_search(chunk.text) for chunk in chunks]
            self._search_matrix = self.vectorizer.fit_transform(raw_texts)
            self._normalized_search_matrix = self._normalized_vectorizer.fit_transform(self._normalized_search_texts)
        else:
            self._search_matrix = None
            self._normalized_search_texts = []
            self._normalized_search_matrix = None

    def match_faq(self, query: str, threshold: float = 0.64) -> FAQEntry | None:
        match, _score = self.match_faq_with_score(query, threshold=threshold)
        return match

    def match_faq_with_score(self, query: str, threshold: float = 0.64) -> tuple[FAQEntry | None, float]:
        faq_questions = [entry.question for entry in self.faq_entries]
        if not faq_questions:
            return None, 0.0

        faq_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        faq_matrix = faq_vectorizer.fit_transform(faq_questions)
        query_vector = faq_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, faq_matrix).flatten()
        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])
        if best_score >= threshold:
            return self.faq_entries[best_index], best_score
        return None, best_score

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_types: set[str] | None = None,
        exclude_source_types: set[str] | None = None,
    ) -> list[KnowledgeChunk]:
        if (
            not self._search_chunks
            or self._search_matrix is None
            or self._normalized_search_matrix is None
        ):
            return []

        query_vector = self.vectorizer.transform([query])
        query_normalized = self._normalize_for_search(query)
        normalized_vector = self._normalized_vectorizer.transform([query_normalized])
        raw_similarities = cosine_similarity(query_vector, self._search_matrix).flatten()
        normalized_similarities = cosine_similarity(normalized_vector, self._normalized_search_matrix).flatten()
        ranked_indexes = self._rank_indexes(
            query,
            query_normalized,
            raw_similarities,
            normalized_similarities,
            passage_texts=[chunk.text for chunk in self._search_chunks],
        )

        results: list[KnowledgeChunk] = []
        for index in ranked_indexes:
            score = round(
                self._combined_score(
                    query,
                    query_normalized,
                    self._search_chunks[index].text,
                    raw_similarities[index],
                    normalized_similarities[index],
                ),
                4,
            )
            if score <= 0:
                continue
            chunk = self._search_chunks[index]
            if source_types and chunk.source_type not in source_types:
                continue
            if exclude_source_types and chunk.source_type in exclude_source_types:
                continue
            results.append(
                KnowledgeChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_name=chunk.source_name,
                    source_type=chunk.source_type,
                    text=chunk.text,
                    score=score,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def has_uploaded_files(self) -> bool:
        return any(source.get("source_type") in UPLOADED_FILE_TYPES for source in self.extra_sources)

    def search_uploaded_files(self, query: str, top_k: int = 5) -> list[KnowledgeChunk]:
        return self.search(query, top_k=top_k, source_types=UPLOADED_FILE_TYPES)

    def get_stats(self) -> dict:
        return {
            "faq_count": len(self.faq_entries),
            "source_count": len(self.extra_sources),
            "chunk_count": len(self._search_chunks),
            "source_names": [source["source_name"] for source in self.extra_sources[-8:]][::-1],
        }

    def get_source_chunks(self, source_name: str, source_type: str, limit: int = 4) -> list[KnowledgeChunk]:
        matches = [
            chunk
            for chunk in self._search_chunks
            if chunk.source_name == source_name and chunk.source_type == source_type
        ]
        return matches[:limit]

    def search_within_source(
        self,
        query: str,
        source_name: str,
        source_type: str,
        top_k: int = 4,
    ) -> list[KnowledgeChunk]:
        source_chunks = self.get_source_chunks(source_name, source_type, limit=1000)
        if not source_chunks:
            return []

        candidate_passages: list[tuple[str, str]] = []
        for chunk in source_chunks:
            passages = self._split_into_passages(chunk.text)
            if passages:
                candidate_passages.extend((passage, chunk.chunk_id) for passage in passages)
            else:
                candidate_passages.append((chunk.text, chunk.chunk_id))

        if not candidate_passages:
            return source_chunks[:top_k]

        passage_texts = [item[0] for item in candidate_passages]
        normalized_passages = [self._normalize_for_search(text) for text in passage_texts]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        normalized_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(3, 5),
            analyzer="char_wb",
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(passage_texts)
        normalized_matrix = normalized_vectorizer.fit_transform(normalized_passages)
        query_vector = vectorizer.transform([query])
        query_normalized = self._normalize_for_search(query)
        normalized_query_vector = normalized_vectorizer.transform([query_normalized])
        raw_similarities = cosine_similarity(query_vector, matrix).flatten()
        normalized_similarities = cosine_similarity(normalized_query_vector, normalized_matrix).flatten()
        ranked_indexes = self._rank_indexes(query, query_normalized, raw_similarities, normalized_similarities, passage_texts=passage_texts)

        results: list[KnowledgeChunk] = []
        seen_texts: set[str] = set()
        for index in ranked_indexes:
            score = round(
                self._combined_score(
                    query,
                    query_normalized,
                    passage_texts[index],
                    raw_similarities[index],
                    normalized_similarities[index],
                ),
                4,
            )
            if score <= 0:
                continue
            text, chunk_id = candidate_passages[index]
            normalized = text.strip()
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            results.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source_id=source_chunks[0].source_id,
                    source_name=source_name,
                    source_type=source_type,
                    text=normalized,
                    score=score,
                )
            )
            if len(results) >= top_k:
                break

        return results or source_chunks[:top_k]

    def ingest_file(self, filename: str, file_bytes: bytes) -> dict:
        extension = Path(filename).suffix.lower()
        source_name = Path(filename).name

        if extension == ".csv":
            texts = self._parse_csv(file_bytes)
            source_type = "csv"
        elif extension == ".json":
            texts = self._parse_json(file_bytes)
            source_type = "json"
        elif extension == ".txt":
            texts = [file_bytes.decode("utf-8", errors="ignore")]
            source_type = "txt"
        elif extension == ".pdf":
            texts = self._parse_pdf(file_bytes)
            source_type = "pdf"
        elif extension == ".docx":
            texts = self._parse_docx(file_bytes)
            source_type = "docx"
        else:
            raise ValueError("Unsupported file type. Use CSV, JSON, TXT, PDF, or DOCX.")

        return self._store_source(source_name=source_name, source_type=source_type, texts=texts)

    def ingest_manual_text(self, title: str, text: str) -> dict:
        clean_title = title.strip() or "Manual Knowledge Note"
        return self._store_source(source_name=clean_title, source_type="manual", texts=[text])

    def ingest_url(self, url: str) -> dict:
        request = Request(url, headers={"User-Agent": "HybridAIKnowledgeBot/1.0"})
        try:
            with urlopen(request, timeout=12) as response:  # noqa: S310
                html = response.read().decode("utf-8", errors="ignore")
        except URLError as error:
            raise ValueError(f"Unable to fetch the website content: {error.reason}") from error

        parser = _HTMLTextExtractor()
        parser.feed(html)
        extracted_text = parser.get_text()
        if not extracted_text.strip():
            raise ValueError("The website was fetched, but no readable text was extracted.")

        return self._store_source(source_name=url, source_type="url", texts=[extracted_text])

    def _store_source(self, source_name: str, source_type: str, texts: list[str]) -> dict:
        combined_chunks: list[dict] = []
        for text in texts:
            for chunk_text in self._chunk_text(text):
                combined_chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "text": chunk_text,
                    }
                )

        if not combined_chunks:
            raise ValueError("No readable text was found in the submitted knowledge source.")

        self.extra_sources = [
            source
            for source in self.extra_sources
            if not (source["source_name"] == source_name and source["source_type"] == source_type)
        ]

        source = {
            "source_id": str(uuid.uuid4()),
            "source_name": source_name,
            "source_type": source_type,
            "chunks": combined_chunks,
        }
        self.extra_sources.append(source)
        self._save_store()
        self._rebuild_index()
        return {
            "source_name": source_name,
            "source_type": source_type,
            "chunks_added": len(combined_chunks),
            "stats": self.get_stats(),
        }

    def _parse_csv(self, file_bytes: bytes) -> list[str]:
        decoded = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(decoded))
        rows = list(reader)
        if not rows:
            return []

        texts: list[str] = []
        if {"question", "answer"}.issubset(reader.fieldnames or []):
            for row in rows:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if question and answer:
                    texts.append(f"Question: {question}\nAnswer: {answer}")
            return texts

        for row in rows:
            fragments = [f"{key}: {value}" for key, value in row.items() if str(value).strip()]
            if fragments:
                texts.append("\n".join(fragments))
        return texts

    def _parse_json(self, file_bytes: bytes) -> list[str]:
        payload = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        results: list[str] = []

        def walk(node, prefix: str = "") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}{key}: ")
            elif isinstance(node, list):
                for item in node:
                    walk(item, prefix)
            else:
                text = f"{prefix}{node}".strip()
                if text:
                    results.append(text)

        walk(payload)
        return results

    def _parse_pdf(self, file_bytes: bytes) -> list[str]:
        if PdfReader is None:
            raise ValueError("PDF support requires the `pypdf` package. Install project dependencies and try again.")

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as error:  # pragma: no cover - parser/library dependent
            raise ValueError(f"The PDF could not be read. Please upload a valid readable PDF. Details: {error}") from error
        return [text for text in pages if text.strip()]

    def _parse_docx(self, file_bytes: bytes) -> list[str]:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            document_xml = archive.read("word/document.xml")

        root = ElementTree.fromstring(document_xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
            joined = "".join(texts).strip()
            if joined:
                paragraphs.append(joined)
        return paragraphs

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 560, overlap: int = 90) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        if len(normalized) <= chunk_size:
            return [normalized]

        chunks = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_size)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks

    @staticmethod
    def _split_into_passages(text: str) -> list[str]:
        if not text.strip():
            return []

        qa_blocks = []
        question_answer_matches = re.findall(
            r"Question:\s*.*?\s*Answer:\s*.*?(?=\s*Question:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if question_answer_matches:
            qa_blocks.extend(re.sub(r"\s+", " ", block).strip(" -") for block in question_answer_matches)

        numbered_qa_matches = re.findall(
            r"Q\d+:\s*.*?\?\s*.*?(?=\s*Q\d+:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if numbered_qa_matches:
            qa_blocks.extend(re.sub(r"\s+", " ", block).strip(" -") for block in numbered_qa_matches)

        if qa_blocks:
            return [block for block in qa_blocks if len(block) >= 10]

        raw_segments = re.split(r"(?<=[.!?])\s+|\n+", text)
        segments: list[str] = []
        for segment in raw_segments:
            cleaned = re.sub(r"\s+", " ", segment).strip(" -")
            if len(cleaned) >= 20:
                segments.append(cleaned)

        if segments:
            return segments

        cleaned = re.sub(r"\s+", " ", text).strip()
        return [cleaned] if cleaned else []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+(?:[+\-*/][a-z0-9]+)?", text.lower())

    @classmethod
    def _expand_terms(cls, text: str) -> str:
        tokens = cls._tokenize(text)
        expanded_terms: list[str] = []
        for token in tokens:
            expanded_terms.append(token)
            expanded_terms.extend(sorted(TERM_SYNONYMS.get(token, set())))
        return " ".join(expanded_terms)

    @classmethod
    def _normalize_for_search(cls, text: str) -> str:
        lowered = text.lower()
        compact_math = re.sub(r"\s*([+\-*/=])\s*", r"\1", lowered)
        expanded = cls._expand_terms(compact_math)
        return f"{compact_math} {expanded}".strip()

    @classmethod
    def _keyword_overlap_score(cls, query_normalized: str, candidate_text: str) -> float:
        query_terms = set(cls._tokenize(query_normalized))
        candidate_terms = set(cls._tokenize(cls._normalize_for_search(candidate_text)))
        if not query_terms or not candidate_terms:
            return 0.0
        overlap = len(query_terms & candidate_terms) / len(query_terms)
        return min(overlap, 1.0)

    @staticmethod
    def _expression_bonus(query_normalized: str, candidate_text: str) -> float:
        expressions = re.findall(r"\b\d+(?:[+\-*/]\d+)+(?:=\d+)?\b", query_normalized)
        if not expressions:
            return 0.0
        candidate_normalized = re.sub(r"\s+", "", candidate_text.lower())
        for expression in expressions:
            if expression in candidate_normalized:
                return 0.4
        return 0.0

    @classmethod
    def _combined_score(
        cls,
        query: str,
        query_normalized: str,
        candidate_text: str,
        raw_score: float,
        normalized_score: float,
    ) -> float:
        keyword_score = cls._keyword_overlap_score(query_normalized, candidate_text)
        exact_bonus = 0.2 if query.lower().strip() in candidate_text.lower() else 0.0
        expression_bonus = cls._expression_bonus(query_normalized, candidate_text)
        return (
            0.45 * float(raw_score)
            + 0.25 * float(normalized_score)
            + 0.25 * keyword_score
            + exact_bonus
            + expression_bonus
        )

    @classmethod
    def _rank_indexes(
        cls,
        query: str,
        query_normalized: str,
        raw_similarities,
        normalized_similarities,
        *,
        passage_texts: list[str] | None = None,
    ):
        texts = passage_texts or []
        scores = []
        total = len(raw_similarities)
        for index in range(total):
            candidate_text = texts[index] if texts else ""
            scores.append(
                (
                    cls._combined_score(
                        query,
                        query_normalized,
                        candidate_text,
                        raw_similarities[index],
                        normalized_similarities[index],
                    ),
                    index,
                )
            )
        scores.sort(key=lambda item: item[0], reverse=True)
        return [index for _score, index in scores]


def chunk_to_dict(chunk: KnowledgeChunk) -> dict:
    return asdict(chunk)
