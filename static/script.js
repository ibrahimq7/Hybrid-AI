const STORAGE_KEY = "hybrid_ai_chats_v3";
const SETTINGS_KEY = "hybrid_ai_settings_v2";

const chatbox = document.getElementById("chatbox");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const clearChatsBtn = document.getElementById("clearChatsBtn");
const attachBtn = document.getElementById("attachBtn");
const knowledgeFile = document.getElementById("knowledgeFile");
const recordBtn = document.getElementById("recordBtn");
const confirmVoiceBtn = document.getElementById("confirmVoiceBtn");
const voiceToggle = document.getElementById("voiceToggle");
const stopVoiceBtn = document.getElementById("stopVoiceBtn");
const modeSelect = document.getElementById("modeSelect");
const assistantStatus = document.getElementById("assistantStatus");
const chatHistoryList = document.getElementById("chatHistoryList");
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = null;
let pendingTranscript = "";
let isRecording = false;
let transcriptFinalized = false;
let isSpeaking = false;
let state = loadState();
let settings = loadSettings();
let activeChatId = state[0]?.id || createChat().id;

function loadState() {
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
        if (Array.isArray(saved) && saved.length) return saved;
    } catch (_error) {}
    return [];
}

function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function loadSettings() {
    try {
        return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{"voiceEnabled":true,"mode":"hybrid"}');
    } catch (_error) {
        return { voiceEnabled: true, mode: "hybrid" };
    }
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function createChat() {
    const chat = {
        id: crypto.randomUUID(),
        title: "New chat",
        updatedAt: new Date().toISOString(),
        messages: [],
    };
    state.unshift(chat);
    saveState();
    return chat;
}

function getActiveChat() {
    return state.find((chat) => chat.id === activeChatId);
}

function escapeHtml(text) {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function relativeTime(dateString) {
    const elapsed = Date.now() - new Date(dateString).getTime();
    const minutes = Math.round(elapsed / 60000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.round(hours / 24)}d ago`;
}

function setStatus(message) {
    assistantStatus.textContent = message;
}

function autoResize() {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 240)}px`;
}

function renderChatHistory() {
    chatHistoryList.innerHTML = "";
    state.forEach((chat) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `history-item ${chat.id === activeChatId ? "active" : ""}`;
        button.innerHTML = `
            <span class="history-title">${escapeHtml(chat.title)}</span>
            <span class="history-time">${relativeTime(chat.updatedAt)}</span>
            <span class="history-delete" data-chat-id="${chat.id}">x</span>
        `;
        button.addEventListener("click", (event) => {
            const deleteTarget = event.target;
            if (deleteTarget.classList.contains("history-delete")) {
                event.stopPropagation();
                deleteChat(deleteTarget.dataset.chatId);
                return;
            }
            activeChatId = chat.id;
            renderChatHistory();
            renderMessages();
        });
        chatHistoryList.appendChild(button);
    });
}

function renderWelcomeState() {
    chatbox.innerHTML = `
        <article class="welcome-card">
            <h3>Hybrid AI is ready.</h3>
            <p>Ask with text or voice, listen to spoken replies when voice output is enabled, and add files or links when you want the assistant to answer from your own knowledge.</p>
            <div class="quick-prompts" aria-label="Example prompts">
                <button type="button" data-prompt="What is artificial intelligence?">What is AI?</button>
                <button type="button" data-prompt="Explain machine learning in simple words.">Explain ML</button>
                <button type="button" data-prompt="Summarize this website: ">Summarize link</button>
            </div>
        </article>
    `;
    chatbox.querySelectorAll("[data-prompt]").forEach((button) => {
        button.addEventListener("click", () => {
            questionInput.value = button.dataset.prompt || "";
            autoResize();
            questionInput.focus();
        });
    });
}

function renderMessages() {
    const chat = getActiveChat();
    if (!chat || !chat.messages.length) {
        renderWelcomeState();
        return;
    }

    chatbox.innerHTML = "";
    chat.messages.forEach((message) => {
        const article = document.createElement("article");
        article.className = `message ${message.role}`;
        const pills = [];
        if (message.mode) pills.push(message.mode.replaceAll("_", " "));
        if (message.confidence) pills.push(`${message.confidence} confidence`);
        if (message.provider) pills.push(message.provider);
        if (message.assistantMode) pills.push(`${message.assistantMode} mode`);

        article.innerHTML = `
            <div class="message-header">
                <span class="message-role">${message.label}</span>
                <span class="message-meta">${new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
            </div>
            <p>${escapeHtml(message.content)}</p>
            ${message.note ? `<div class="message-note">${escapeHtml(message.note)}</div>` : ""}
            ${pills.length ? `<div class="meta-row">${pills.map((pill) => `<span class="meta-pill">${escapeHtml(pill)}</span>`).join("")}</div>` : ""}
        `;
        chatbox.appendChild(article);
    });
    chatbox.scrollTop = chatbox.scrollHeight;
}

function buildAnswerNote(payload) {
    const sourceTypes = new Set((payload.sources || []).map((source) => source.source_type));

    if (payload.mode === "strict_faq") {
        return "Answered directly from the trusted FAQ dataset.";
    }
    if (payload.assistant_mode === "strict" && sourceTypes.size) {
        const uploadedTypes = ["csv", "json", "txt", "pdf", "docx"].filter((type) => sourceTypes.has(type));
        if (uploadedTypes.length) {
            return `Strict mode answered from your uploaded ${uploadedTypes[0].toUpperCase()} knowledge.`;
        }
        return "Strict mode answered only from your stored knowledge.";
    }
    if (sourceTypes.has("csv") || sourceTypes.has("json") || sourceTypes.has("txt") || sourceTypes.has("pdf") || sourceTypes.has("docx")) {
        if (payload.generator_provider) {
            return "Matched your uploaded file and refined the answer with Groq.";
        }
        return "Answered from matched uploaded file knowledge.";
    }
    if (payload.generator_provider) {
        return `Generated in hybrid mode with ${payload.generator_provider}.`;
    }
    return payload.guidance || "";
}

function addMessage(message) {
    const chat = getActiveChat();
    if (!chat) return;
    chat.messages.push(message);
    chat.updatedAt = new Date().toISOString();
    if (chat.title === "New chat" && message.role === "user") {
        chat.title = message.content.slice(0, 42) || "New chat";
    }
    saveState();
    renderChatHistory();
    renderMessages();
}

function deleteChat(chatId) {
    state = state.filter((chat) => chat.id !== chatId);
    if (!state.length) {
        activeChatId = createChat().id;
    } else if (activeChatId === chatId) {
        activeChatId = state[0].id;
    }
    saveState();
    renderChatHistory();
    renderMessages();
}

function clearAllChats() {
    state = [];
    activeChatId = createChat().id;
    saveState();
    renderChatHistory();
    renderMessages();
}

function showTypingIndicator() {
    const article = document.createElement("article");
    article.id = "typingIndicator";
    article.className = "message assistant";
    article.innerHTML = `
        <div class="message-header">
            <span class="message-role">${settings.mode === "strict" ? "Strict AI" : "Hybrid AI"}</span>
            <span class="message-meta">Thinking</span>
        </div>
        <div class="typing"><span></span><span></span><span></span></div>
    `;
    chatbox.appendChild(article);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function hideTypingIndicator() {
    document.getElementById("typingIndicator")?.remove();
}

function buildHistoryForApi(chat) {
    return chat.messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .slice(-8)
        .map((message) => ({
            role: message.role === "user" ? "user" : "assistant",
            content: message.content,
        }));
}

function speakText(text) {
    if (!settings.voiceEnabled || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 1.04;
    utterance.pitch = 0.95;
    utterance.onstart = () => {
        isSpeaking = true;
        stopVoiceBtn.disabled = false;
        setStatus("Reading the answer aloud. Click Stop voice to stop playback.");
    };
    utterance.onend = () => {
        isSpeaking = false;
        stopVoiceBtn.disabled = true;
        setStatus("Response delivered.");
    };
    utterance.onerror = () => {
        isSpeaking = false;
        stopVoiceBtn.disabled = true;
        setStatus("Voice playback stopped.");
    };
    window.speechSynthesis.speak(utterance);
}

function stopSpeaking() {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    isSpeaking = false;
    stopVoiceBtn.disabled = true;
    setStatus("Voice playback stopped.");
}

function syncControls() {
    modeSelect.value = settings.mode;
    voiceToggle.classList.toggle("active", settings.voiceEnabled);
    voiceToggle.textContent = settings.voiceEnabled ? "Voice replies on" : "Voice replies off";
    voiceToggle.setAttribute("aria-pressed", String(settings.voiceEnabled));
    recordBtn.textContent = isRecording ? "Stop" : "Mic";
    recordBtn.classList.toggle("recording", isRecording);
    confirmVoiceBtn.classList.toggle("hidden", !pendingTranscript);
    stopVoiceBtn.disabled = !isSpeaking;
}

function isVoiceSupported() {
    return Boolean(SpeechRecognition) && (window.isSecureContext || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
}

async function askQuestion(prefilledQuestion) {
    const question = (prefilledQuestion || questionInput.value).trim();
    if (!question) return;

    addMessage({
        role: "user",
        label: "You",
        content: question,
        createdAt: new Date().toISOString(),
    });

    stopSpeaking();
    questionInput.value = "";
    autoResize();
    pendingTranscript = "";
    syncControls();
    setStatus("Searching your data and preparing the reply.");
    sendBtn.disabled = true;
    showTypingIndicator();

    try {
        const chat = getActiveChat();
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                history: buildHistoryForApi(chat),
                mode: settings.mode,
            }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "The assistant could not process the request.");

        hideTypingIndicator();
        addMessage({
            role: "assistant",
            label: payload.assistant_mode === "strict" ? "Strict AI" : "Hybrid AI",
            content: payload.answer,
            mode: payload.mode,
            confidence: payload.confidence,
            provider: payload.generator_provider,
            assistantMode: payload.assistant_mode,
            note: buildAnswerNote(payload),
            createdAt: new Date().toISOString(),
        });
        speakText(payload.answer);
        setStatus(payload.generator_provider ? `Response delivered by ${payload.generator_provider}.` : "Response delivered.");
    } catch (error) {
        hideTypingIndicator();
        addMessage({
            role: "system",
            label: "System",
            content: error.message,
            createdAt: new Date().toISOString(),
        });
        setStatus("There was a problem completing that request.");
    } finally {
        sendBtn.disabled = false;
    }
}

async function uploadKnowledgeFile(file) {
    const formData = new FormData();
    formData.append("knowledge_file", file);

    addMessage({
        role: "system",
        label: "System",
        content: `Uploading ${file.name} into your knowledge base...`,
        createdAt: new Date().toISOString(),
    });

    try {
        const response = await fetch("/api/knowledge", {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "File upload failed.");

        addMessage({
            role: "assistant",
            label: "Hybrid AI",
            content: `${file.name} was added successfully. I can use it in strict retrieval and hybrid answers now.`,
            createdAt: new Date().toISOString(),
        });
        setStatus(`${file.name} added to the knowledge base.`);
    } catch (error) {
        addMessage({
            role: "system",
            label: "System",
            content: error.message,
            createdAt: new Date().toISOString(),
        });
        setStatus("File upload failed.");
    } finally {
        knowledgeFile.value = "";
    }
}

function initializeSpeechRecognition() {
    if (!SpeechRecognition) {
        recordBtn.disabled = true;
        recordBtn.textContent = "No mic";
        setStatus("Voice input needs a browser with speech recognition support.");
        return;
    }
    if (!isVoiceSupported()) {
        recordBtn.disabled = true;
        recordBtn.textContent = "No mic";
        setStatus("Voice input needs HTTPS or localhost plus microphone permission.");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    recognition.onstart = () => {
        isRecording = true;
        transcriptFinalized = false;
        syncControls();
        setStatus("Recording voice. Click Stop when you finish speaking.");
    };

    recognition.onresult = (event) => {
        let finalTranscript = "";
        let interimTranscript = "";
        for (let index = 0; index < event.results.length; index += 1) {
            const result = event.results[index];
            if (result.isFinal) {
                finalTranscript += result[0].transcript + " ";
            } else {
                interimTranscript += result[0].transcript + " ";
            }
        }
        pendingTranscript = `${finalTranscript}${interimTranscript}`.trim();
        transcriptFinalized = finalTranscript.trim().length > 0;
        questionInput.value = pendingTranscript;
        autoResize();
        setStatus(transcriptFinalized ? "Voice captured. Review it or click Use voice text." : "Listening... your speech is being converted to text.");
        syncControls();
    };

    recognition.onerror = (event) => {
        isRecording = false;
        syncControls();
        const messages = {
            "not-allowed": "Microphone access was denied. Allow mic permission and try again.",
            "audio-capture": "No microphone was detected. Check your device and browser settings.",
            "network": "Speech recognition network access failed. Try again in Chrome with internet access.",
            "no-speech": "No speech was detected. Try again and speak a little closer to the mic.",
        };
        setStatus(messages[event.error] || "Voice input failed. You can still type your message.");
    };

    recognition.onend = () => {
        isRecording = false;
        syncControls();
        if (pendingTranscript) {
            setStatus("Recording stopped. Review the transcript, then press Send.");
        }
    };
}

function startNewChat() {
    const chat = createChat();
    activeChatId = chat.id;
    saveState();
    renderChatHistory();
    renderMessages();
    setStatus("Fresh chat started.");
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion();
});

questionInput.addEventListener("input", autoResize);
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
});

newChatBtn.addEventListener("click", startNewChat);
clearChatsBtn.addEventListener("click", clearAllChats);

attachBtn.addEventListener("click", () => knowledgeFile.click());
knowledgeFile.addEventListener("change", () => {
    if (knowledgeFile.files[0]) {
        uploadKnowledgeFile(knowledgeFile.files[0]);
    }
});

recordBtn.addEventListener("click", () => {
    if (!recognition) {
        setStatus("Voice input is unavailable in this browser.");
        return;
    }
    if (isRecording) {
        recognition.stop();
        return;
    }
    pendingTranscript = "";
    transcriptFinalized = false;
    questionInput.value = "";
    autoResize();
    syncControls();
    recognition.start();
});

confirmVoiceBtn.addEventListener("click", () => {
    if (!pendingTranscript) return;
    questionInput.value = pendingTranscript;
    autoResize();
    questionInput.focus();
    setStatus("Voice converted to text. Review it and press Send when ready.");
});

voiceToggle.addEventListener("click", () => {
    settings.voiceEnabled = !settings.voiceEnabled;
    saveSettings();
    if (!settings.voiceEnabled) {
        stopSpeaking();
    }
    syncControls();
    setStatus(settings.voiceEnabled ? "Voice replies are enabled." : "Voice replies are disabled.");
});

stopVoiceBtn.addEventListener("click", () => {
    stopSpeaking();
});

modeSelect.addEventListener("change", () => {
    settings.mode = modeSelect.value;
    saveSettings();
    setStatus(settings.mode === "strict" ? "Strict mode is active." : "Hybrid mode is active.");
});

initializeSpeechRecognition();
syncControls();
renderChatHistory();
renderMessages();
autoResize();
