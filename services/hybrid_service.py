from __future__ import annotations

import re

from knowledge_base import KnowledgeBase, chunk_to_dict
from services.groq_service import GroqService


URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
UPLOADED_FILE_REQUEST_PATTERN = re.compile(
    r"\b(uploaded file|uploaded files|upload file|upload files|document|documents|pdf|csv|word file|docx|from file)\b",
    re.IGNORECASE,
)
FILE_REQUEST_CLEANUP_PATTERN = re.compile(
    r"\b(give me answer for this question from the uploaded file|answer from the uploaded file|from the uploaded file|from uploaded file|from the file|from this file|using the uploaded file|using this file|uploaded file|uploaded files|document|documents|pdf|csv|word file|docx)\b",
    re.IGNORECASE,
)


class HybridAssistantService:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.groq = GroqService()
        self.last_provider_error: str | None = None
        self.last_provider_name: str | None = None

    def ingest_file(self, filename: str, file_bytes: bytes) -> dict:
        result = self.knowledge_base.ingest_file(filename, file_bytes)
        self.knowledge_base.reload()
        return result

    def ingest_manual_text(self, title: str, text: str) -> dict:
        result = self.knowledge_base.ingest_manual_text(title, text)
        self.knowledge_base.reload()
        return result

    def ingest_url(self, url: str) -> dict:
        result = self.knowledge_base.ingest_url(url)
        self.knowledge_base.reload()
        return result

    def answer_question(self, user_question: str, history: list[dict] | None = None, mode: str = "hybrid") -> dict:
        cleaned_query = user_question.strip()
        safe_mode = mode if mode in {"strict", "hybrid"} else "hybrid"
        history = history or []

        detected_url = self._extract_url(cleaned_query)
        if detected_url:
            return self._handle_url_query(cleaned_query, detected_url, history, safe_mode)

        file_focused = self._is_uploaded_file_request(cleaned_query)
        retrieval_query = self._normalize_retrieval_query(cleaned_query, file_focused=file_focused)
        faq_answer = None
        faq_score = 0.0
        if not file_focused:
            faq_answer, faq_score = self.knowledge_base.match_faq_with_score(cleaned_query)
        retrieved_chunks = self._search_relevant_chunks(
            retrieval_query,
            file_focused=file_focused,
            prefer_uploaded=safe_mode == "strict",
        )

        uploaded_match = self._top_uploaded_match(retrieval_query)
        should_prefer_uploaded = self._should_prefer_uploaded_answer(
            uploaded_match,
            faq_score,
            file_focused=file_focused,
            mode=safe_mode,
        )

        if should_prefer_uploaded and uploaded_match:
            uploaded_chunks = self._get_grounded_source_matches(retrieval_query, uploaded_match, matches=retrieved_chunks)
            if safe_mode == "strict":
                return self._build_response(
                    cleaned_query,
                    self._compose_grounded_answer(cleaned_query, uploaded_chunks),
                    "strict_retrieval",
                    "medium",
                    f"Strict mode answered directly from your uploaded {uploaded_match.source_type.upper()} knowledge.",
                    None,
                    uploaded_chunks,
                    safe_mode,
                )

        if faq_answer:
            return self._build_response(
                cleaned_query,
                faq_answer.answer,
                "strict_faq",
                "high",
                "Answered directly from the trusted FAQ dataset.",
                None,
                retrieved_chunks,
                safe_mode,
            )

        if safe_mode == "strict":
            strict_threshold = 0.14 if file_focused else 0.18
            if retrieved_chunks and retrieved_chunks[0].score >= strict_threshold:
                grounded_chunks = self._get_grounded_source_matches(retrieval_query, retrieved_chunks[0], matches=retrieved_chunks)
                return self._build_response(
                    cleaned_query,
                    self._compose_grounded_answer(cleaned_query, grounded_chunks),
                    "strict_retrieval",
                    "medium",
                    "Strict mode used only your FAQ and matched uploaded knowledge.",
                    None,
                    grounded_chunks,
                    safe_mode,
                )
            return self._build_response(
                cleaned_query,
                "Strict mode could not find a reliable answer in your FAQ or uploaded knowledge.",
                "strict_no_match",
                "low",
                "Switch to hybrid mode if you want Groq to help when your data is insufficient.",
                None,
                retrieved_chunks,
                safe_mode,
            )

        hybrid_threshold = 0.12 if file_focused else 0.2
        if retrieved_chunks and retrieved_chunks[0].score >= hybrid_threshold:
            context_chunks = self._get_grounded_source_matches(retrieval_query, retrieved_chunks[0], matches=retrieved_chunks)
            generated = self._generate_with_context(cleaned_query, context_chunks, history, file_focused=file_focused)
            if generated:
                return self._build_response(
                    cleaned_query,
                    generated,
                    "rag_generation",
                    "medium",
                    "Answered using your retrieved knowledge plus Groq.",
                    self.last_provider_name,
                    context_chunks,
                    safe_mode,
                )
            return self._build_response(
                cleaned_query,
                self._compose_grounded_answer(cleaned_query, context_chunks),
                "retrieval_grounded",
                "medium",
                "Used retrieved knowledge directly because Groq is unavailable.",
                None,
                context_chunks,
                safe_mode,
            )

        general = self._generate_general_answer(cleaned_query, history)
        if general:
            return self._build_response(
                cleaned_query,
                general,
                "generative_fallback",
                "medium",
                "No strong knowledge match was found, so Groq handled the request.",
                self.last_provider_name,
                retrieved_chunks,
                safe_mode,
            )

        return self._build_response(
            cleaned_query,
            self._build_provider_unavailable_message(),
            "no_match",
            "low",
            "Knowledge-first search found no reliable answer.",
            None,
            retrieved_chunks,
            safe_mode,
        )

    def _handle_url_query(self, query: str, url: str, history: list[dict], mode: str) -> dict:
        ingest_result = self.ingest_url(url)
        source_chunks = self.knowledge_base.get_source_chunks(url, "url", limit=4)
        prompt_text = query.replace(url, "").strip()
        wants_summary = not prompt_text or any(word in query.lower() for word in ["summary", "summarize", "summarise", "explain", "tell me"])

        if wants_summary:
            if mode == "hybrid":
                generated = self._generate_summary_from_chunks(url, source_chunks, history)
                if generated:
                    answer = generated
                else:
                    answer = self._compose_grounded_answer(source_chunks)
            else:
                answer = self._compose_grounded_answer(source_chunks)
        else:
            answer = self.answer_question(prompt_text, history=history, mode=mode)["answer"]

        return {
            "question": query,
            "answer": answer,
            "mode": "url_summary" if wants_summary else "url_ingested",
            "confidence": "medium",
            "guidance": f"Fetched and indexed website content from {url}.",
            "generator_provider": self.last_provider_name if wants_summary else None,
            "assistant_mode": mode,
            "knowledge_event": ingest_result,
            "sources": [chunk_to_dict(chunk) for chunk in source_chunks],
        }

    def _generate_with_context(self, question: str, chunks, history: list[dict], *, file_focused: bool = False) -> str | None:
        context = "\n\n".join(f"- {chunk.text}" for chunk in chunks[:4])
        conversation = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '').strip()}"
            for item in history[-6:]
            if item.get("content")
        )
        mode_instruction = (
            "The user explicitly wants the answer from uploaded files, so prioritize the retrieved uploaded-file context."
            if file_focused
            else "Use the provided knowledge context first."
        )
        prompt = (
            "You are Hybrid AI, a polished professional chatbot.\n"
            f"{mode_instruction} If context is enough, answer from it clearly.\n"
            "If context is partial, answer naturally but do not invent that the data said more than it did.\n\n"
            f"Conversation history:\n{conversation or 'No previous conversation.'}\n\n"
            f"Knowledge context:\n{context}\n\n"
            f"Question: {question}"
        )
        result = self._generate_llm(prompt)
        return result.text

    def _generate_general_answer(self, question: str, history: list[dict]) -> str | None:
        conversation = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '').strip()}"
            for item in history[-6:]
            if item.get("content")
        )
        prompt = (
            "You are Hybrid AI, a professional assistant with a calm voice-friendly style. "
            "Answer quickly, accurately, and efficiently. Use simple language, avoid unnecessary length, "
            "and format important steps clearly.\n\n"
            f"Conversation history:\n{conversation or 'No previous conversation.'}\n\n"
            f"Question: {question}"
        )
        result = self._generate_llm(prompt)
        return result.text

    def _generate_summary_from_chunks(self, url: str, chunks, history: list[dict]) -> str | None:
        context = "\n\n".join(f"- {chunk.text}" for chunk in chunks[:4])
        conversation = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '').strip()}"
            for item in history[-4:]
            if item.get("content")
        )
        prompt = (
            "Summarize the following webpage content in a clean, intelligent way. "
            "Highlight the main ideas and keep it easy to understand.\n\n"
            f"Conversation history:\n{conversation or 'No previous conversation.'}\n\n"
            f"Source URL: {url}\n"
            f"Content:\n{context}"
        )
        result = self._generate_llm(prompt)
        return result.text

    def _generate_llm(self, prompt: str):
        system_prompt = (
            "You are Hybrid AI, a fast and accurate assistant. "
            "Be polite, professional, direct, and useful. If the answer depends on provided context, "
            "use that context first. Never pretend to know private project data that was not supplied."
        )
        result = self.groq.generate(prompt, system_prompt=system_prompt)
        self.last_provider_error = result.error
        self.last_provider_name = result.provider
        return result

    @staticmethod
    def _compose_grounded_answer(question: str, matches) -> str:
        if not matches:
            return "I could not extract enough readable information from that source."
        lead = HybridAssistantService._extract_precise_answer(question, matches[0].text) or matches[0].text.strip()
        supporting = []
        for match in matches[1:]:
            if not match.text:
                continue
            extracted = HybridAssistantService._extract_precise_answer(question, match.text) or match.text.strip()
            if extracted and extracted != lead:
                supporting.append(extracted)
        if not supporting:
            return lead
        if HybridAssistantService._wants_detailed_answer(question):
            return f"{lead}\n\nRelated detail: {supporting[0]}"
        return lead

    @staticmethod
    def _extract_url(text: str) -> str | None:
        match = URL_PATTERN.search(text)
        return match.group(0) if match else None

    def _build_provider_unavailable_message(self) -> str:
        if self.last_provider_error:
            return (
                "Hybrid mode could not reach Groq right now. "
                f"{self.last_provider_error}"
            )
        return "Hybrid mode could not reach Groq right now."

    def _search_relevant_chunks(self, query: str, *, file_focused: bool, prefer_uploaded: bool = False) -> list:
        if prefer_uploaded and self.knowledge_base.has_uploaded_files():
            uploaded_matches = self.knowledge_base.search_uploaded_files(query)
            if uploaded_matches and uploaded_matches[0].score >= 0.12:
                return uploaded_matches

        if file_focused and self.knowledge_base.has_uploaded_files():
            file_matches = self.knowledge_base.search_uploaded_files(query)
            if file_matches:
                return file_matches

        general_matches = self.knowledge_base.search(query)
        if general_matches:
            return general_matches

        if self.knowledge_base.has_uploaded_files():
            return self.knowledge_base.search_uploaded_files(query)

        return []

    def _top_uploaded_match(self, query: str):
        if not self.knowledge_base.has_uploaded_files():
            return None
        uploaded_matches = self.knowledge_base.search_uploaded_files(query, top_k=1)
        return uploaded_matches[0] if uploaded_matches else None

    @staticmethod
    def _get_chunks_from_same_source(primary_match, matches=None):
        matches = matches or []
        same_source = [
            match
            for match in matches
            if match.source_name == primary_match.source_name and match.source_type == primary_match.source_type
        ]
        if same_source:
            return same_source[:4]
        return [primary_match]

    def _get_grounded_source_matches(self, query: str, primary_match, matches=None):
        focused_matches = self.knowledge_base.search_within_source(
            query,
            primary_match.source_name,
            primary_match.source_type,
            top_k=4,
        )
        if focused_matches:
            return focused_matches
        return self._get_chunks_from_same_source(primary_match, matches=matches)

    @staticmethod
    def _wants_detailed_answer(question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in {"detail", "details", "explain", "policy", "process", "information"})

    @staticmethod
    def _extract_precise_answer(question: str, text: str) -> str | None:
        compact_question = re.sub(r"\s+", " ", question).strip()
        compact_text = re.sub(r"\s+", " ", text).strip()

        qa_patterns = [
            re.compile(
                r"Question:\s*(?P<question>.*?)\s*Answer:\s*(?P<answer>.*?)(?=\s*Question:|\Z)",
                re.IGNORECASE,
            ),
            re.compile(
                r"Q\d+:\s*(?P<question>.*?\?)\s*(?P<answer>.*?)(?=\s*Q\d+:|\Z)",
                re.IGNORECASE,
            ),
        ]

        normalized_question = HybridAssistantService._normalize_compare_text(compact_question)
        best_answer = None
        best_score = 0.0

        for pattern in qa_patterns:
            for match in pattern.finditer(compact_text):
                candidate_question = (match.group("question") or "").strip(" :-")
                candidate_answer = (match.group("answer") or "").strip(" :-")
                score = HybridAssistantService._question_similarity(normalized_question, candidate_question)
                if score > best_score and candidate_answer:
                    best_score = score
                    best_answer = candidate_answer

        if best_answer and best_score >= 0.45:
            return best_answer

        sentences = [segment.strip(" :-") for segment in re.split(r"(?<=[.!?])\s+|\s{2,}", compact_text) if segment.strip()]
        if not sentences:
            return None

        best_sentence = None
        best_sentence_score = 0.0
        for sentence in sentences:
            score = HybridAssistantService._question_similarity(normalized_question, sentence)
            if score > best_sentence_score:
                best_sentence_score = score
                best_sentence = sentence

        if best_sentence and best_sentence_score >= 0.35:
            return best_sentence

        return None

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        lowered = text.lower()
        lowered = lowered.replace("×", "*").replace("÷", "/")
        lowered = re.sub(r"\s*([+\-*/=])\s*", r" \1 ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _question_similarity(question: str, candidate: str) -> float:
        normalized_candidate = HybridAssistantService._normalize_compare_text(candidate)
        question_tokens = set(re.findall(r"[a-z0-9+\-*/=]+", question))
        candidate_tokens = set(re.findall(r"[a-z0-9+\-*/=]+", normalized_candidate))
        if not question_tokens or not candidate_tokens:
            return 0.0
        overlap = len(question_tokens & candidate_tokens) / len(question_tokens)
        compact_question = question.replace(" ", "")
        compact_candidate = normalized_candidate.replace(" ", "")
        expression_bonus = 0.4 if compact_question and compact_question in compact_candidate else 0.0
        return min(overlap + expression_bonus, 1.0)

    @staticmethod
    def _should_prefer_uploaded_answer(uploaded_match, faq_score: float, *, file_focused: bool, mode: str) -> bool:
        if not uploaded_match:
            return False
        if file_focused:
            return uploaded_match.score >= 0.1
        if mode == "strict":
            return uploaded_match.score >= max(0.16, faq_score - 0.03)
        return uploaded_match.score >= max(0.14, faq_score)

    @staticmethod
    def _is_uploaded_file_request(text: str) -> bool:
        return bool(UPLOADED_FILE_REQUEST_PATTERN.search(text))

    @staticmethod
    def _normalize_retrieval_query(text: str, *, file_focused: bool) -> str:
        if not file_focused:
            return text
        normalized = FILE_REQUEST_CLEANUP_PATTERN.sub(" ", text)
        normalized = re.sub(r"\s+", " ", normalized).strip(" :,-")
        return normalized or text

    @staticmethod
    def _build_response(question: str, answer: str, mode: str, confidence: str, guidance: str, provider: str | None, chunks, assistant_mode: str) -> dict:
        return {
            "question": question,
            "answer": answer,
            "mode": mode,
            "confidence": confidence,
            "guidance": guidance,
            "generator_provider": provider,
            "assistant_mode": assistant_mode,
            "sources": [chunk_to_dict(chunk) for chunk in chunks[:4]],
        }
