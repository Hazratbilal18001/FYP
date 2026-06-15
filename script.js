/**
 * script.js — Healthcare AI Chatbot Frontend Logic
 *
 * Handles:
 *   - Chat creation, switching, deletion
 *   - Sending messages via SSE (Server-Sent Events)
 *   - Streaming AI responses with typing effect
 *   - PDF download of AI responses (jsPDF)
 *   - Markdown rendering (marked.js)
 *   - Responsive sidebar toggle
 */

// ================================================================
// STATE
// ================================================================

const state = {
    currentChatId: null,
    currentDomain: "medical",
    isStreaming: false,
};

// ================================================================
// DOM ELEMENTS
// ================================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const sidebar        = $("#sidebar");
const sidebarToggle  = $("#sidebarToggle");
const sidebarOverlay = $("#sidebarOverlay");
const chatList       = $("#chatList");
const btnNewChat     = $("#btnNewChat");
const welcomeScreen  = $("#welcomeScreen");
const chatArea       = $("#chatArea");
const messagesContainer = $("#messagesContainer");
const messageInput   = $("#messageInput");
const btnSend        = $("#btnSend");
const domainPills    = $$(".domain-pill");

// ================================================================
// INITIALIZE
// ================================================================

document.addEventListener("DOMContentLoaded", () => {
    loadChatList();
    setupEventListeners();
    configureMarked();
    checkPdfLibrary();
});

/** Configure marked.js for safe Markdown rendering */
function configureMarked() {
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

/** Ensure jsPDF is available and log a helpful warning if not */
function checkPdfLibrary() {
    if (!window.jspdf || !window.jspdf.jsPDF) {
        console.warn("jsPDF is not available. PDF download will not work until the library is loaded.");
    }
}

// ================================================================
// EVENT LISTENERS
// ================================================================

function setupEventListeners() {
    // New chat
    btnNewChat.addEventListener("click", createNewChat);

    // Send message
    btnSend.addEventListener("click", sendMessage);

    // Textarea: enter to send, shift+enter for newline
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea & enable/disable send button
    messageInput.addEventListener("input", () => {
        autoResizeTextarea();
        btnSend.disabled = messageInput.value.trim().length === 0 || state.isStreaming;
    });

    // Domain pills
    domainPills.forEach((pill) => {
        pill.addEventListener("click", () => {
            updateActiveDomain(pill.dataset.domain, true);
        });
    });

    // Welcome cards — clicking sends the prompt
    $$(".welcome-card").forEach((card) => {
        card.addEventListener("click", () => {
            const prompt = card.dataset.prompt;
            const domain = card.dataset.domain;
            if (domain) {
                updateActiveDomain(domain, true);
            }
            if (prompt) {
                messageInput.value = prompt;
                autoResizeTextarea();
                btnSend.disabled = false;
                sendMessage();
            }
        });
    });

    // Sidebar toggle (mobile)
    sidebarToggle.addEventListener("click", toggleSidebar);
    sidebarOverlay.addEventListener("click", closeSidebar);
}

/** Auto-resize textarea to fit content */
function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + "px";
}

// ================================================================
// SIDEBAR
// ================================================================

function toggleSidebar() {
    sidebar.classList.toggle("open");
    sidebarOverlay.classList.toggle("visible");
}

function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("visible");
}

// ================================================================
// CHAT LIST
// ================================================================

/** Load all chats from the API and render them */
async function loadChatList() {
    try {
        const res = await fetch("/api/chats");
        const chats = await res.json();
        renderChatList(chats);
    } catch (err) {
        console.error("Failed to load chats:", err);
    }
}

/** Render the chat list in the sidebar */
function renderChatList(chats) {
    if (!chats.length) {
        chatList.innerHTML = `<div class="chat-list-empty">No chats yet.<br>Start a new one!</div>`;
        return;
    }

    chatList.innerHTML = chats
        .map((chat) => {
            const isActive = chat.chat_id === state.currentChatId;
            const title = chat.title.length > 30 ? chat.title.slice(0, 30) + "…" : chat.title;
            return `
                <div class="chat-item ${isActive ? "active" : ""}" data-chat-id="${chat.chat_id}" data-domain="${chat.domain}">
                    <span class="chat-item-icon">💬</span>
                    <span class="chat-item-title">${escapeHtml(title)}</span>
                    <span class="chat-item-domain">${escapeHtml(chat.domain)}</span>
                    <button class="chat-item-delete" data-delete-id="${chat.chat_id}" title="Delete chat">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    </button>
                </div>
            `;
        })
        .join("");

    // Attach click handlers
    $$(".chat-item").forEach((item) => {
        item.addEventListener("click", (e) => {
            // Don't switch chat if delete button was clicked
            if (e.target.closest(".chat-item-delete")) return;
            const chatId = parseInt(item.dataset.chatId);
            const domain = item.dataset.domain;
            switchToChat(chatId, domain);
        });
    });

    $$(".chat-item-delete").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const chatId = parseInt(btn.dataset.deleteId);
            deleteChat(chatId);
        });
    });
}

// ================================================================
// CHAT OPERATIONS
// ================================================================

/** Create a new empty chat */
async function createNewChat() {
    state.currentChatId = null;
    messagesContainer.innerHTML = "";
    showWelcomeScreen();
    closeSidebar();
    loadChatList();
    messageInput.focus();
}

/** Switch to an existing chat and load its messages */
async function switchToChat(chatId, domain) {
    state.currentChatId = chatId;
    if (domain) {
        updateActiveDomain(domain, false);
    }

    try {
        const res = await fetch(`/api/chats/${chatId}/messages`);
        const messages = await res.json();

        messagesContainer.innerHTML = "";
        messages.forEach((msg) => {
            appendMessage(msg.role === "user" ? "user" : "assistant", msg.content, false);
        });

        showChatArea();
        scrollToBottom();
        loadChatList();
        closeSidebar();
    } catch (err) {
        console.error("Failed to load messages:", err);
    }
}

/** Delete a chat */
async function deleteChat(chatId) {
    try {
        await fetch(`/api/chats/${chatId}`, { method: "DELETE" });

        // If we deleted the active chat, go back to welcome
        if (state.currentChatId === chatId) {
            state.currentChatId = null;
            messagesContainer.innerHTML = "";
            showWelcomeScreen();
        }

        loadChatList();
    } catch (err) {
        console.error("Failed to delete chat:", err);
    }
}

// ================================================================
// SEND MESSAGE & STREAM RESPONSE
// ================================================================

/** Send the current input message to the backend */
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || state.isStreaming) return;

    state.isStreaming = true;
    btnSend.disabled = true;

    // Clear input
    messageInput.value = "";
    messageInput.style.height = "auto";

    // Show chat area if on welcome screen
    showChatArea();

    // Render user message
    appendMessage("user", text, false);
    scrollToBottom();

    // Show typing indicator
    const typingEl = appendTypingIndicator();
    scrollToBottom();

    try {
        // Send to API via SSE
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                chat_id: state.currentChatId,
                domain: state.currentDomain,
            }),
        });

        // Remove typing indicator
        typingEl.remove();

        // Create assistant message element (empty, to be filled by stream)
        const { messageEl, contentEl } = appendMessage("assistant", "", true);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;

                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === "meta") {
                        // Got the chat_id
                        state.currentChatId = data.chat_id;
                        if (data.domain) {
                            updateActiveDomain(data.domain, false);
                        }
                    } else if (data.type === "token") {
                        fullResponse += data.content;
                        // Render markdown progressively
                        contentEl.innerHTML = marked.parse(fullResponse) + `<span class="streaming-cursor">▌</span>`;
                        scrollToBottom();
                    } else if (data.type === "done") {
                        // Some backends send the final response here even if token stream was partial.
                        if (!fullResponse && typeof data.full_response === "string") {
                            fullResponse = data.full_response;
                        }
                        // Final render without cursor
                        contentEl.innerHTML = marked.parse(fullResponse);
                        // Add PDF download button
                        addPdfButton(messageEl, fullResponse);
                    } else if (data.type === "error") {
                        contentEl.innerHTML = `<p style="color: #f87171;">⚠️ Error: ${escapeHtml(data.content)}</p>`;
                    }
                } catch (parseErr) {
                    // Ignore malformed SSE lines
                }
            }
        }

        // Safety: ensure final render if "done" event was missed
        if (fullResponse && !messageEl.querySelector(".btn-download-pdf")) {
            contentEl.innerHTML = marked.parse(fullResponse);
            addPdfButton(messageEl, fullResponse);
        }

        // Refresh chat list (new chat may have been created)
        loadChatList();

    } catch (err) {
        typingEl.remove();
        appendMessage("assistant", `⚠️ Failed to get response. Please check that the server is running.\n\nError: ${err.message}`, false);
        console.error("Stream error:", err);
    } finally {
        state.isStreaming = false;
        btnSend.disabled = messageInput.value.trim().length === 0;
        scrollToBottom();
    }
}

// ================================================================
// MESSAGE RENDERING
// ================================================================

/**
 * Append a message to the chat area.
 * @param {string} role - "user" or "assistant"
 * @param {string} content - message text
 * @param {boolean} isStreaming - if true, returns refs for live-updating
 * @returns {{ messageEl: HTMLElement, contentEl: HTMLElement }}
 */
function appendMessage(role, content, isStreaming = false) {
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;

    const avatar = role === "user" ? "👤" : "🩺";
    const roleName = role === "user" ? "You" : "HealthAI";

    const renderedContent = isStreaming ? "" : (role === "assistant" ? marked.parse(content) : escapeHtml(content));

    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-body">
            <div class="message-role">${roleName}</div>
            <div class="message-content">${renderedContent}</div>
        </div>
    `;

    messagesContainer.appendChild(messageEl);

    const contentEl = messageEl.querySelector(".message-content");

    // Add PDF button for non-streaming assistant messages
    if (role === "assistant" && !isStreaming && content) {
        addPdfButton(messageEl, content);
    }

    return { messageEl, contentEl };
}

/** Add typing indicator */
function appendTypingIndicator() {
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = `
        <div class="message-avatar">🩺</div>
        <div class="message-body">
            <div class="message-role">HealthAI</div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(el);
    return el;
}

// ================================================================
// PDF DOWNLOAD
// ================================================================

/** Add a PDF download button to an assistant message */
function addPdfButton(messageEl, responseText) {
    const body = messageEl.querySelector(".message-body");

    // Don't add if it already exists
    if (body.querySelector(".message-actions")) return;

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "message-actions";

    actionsDiv.innerHTML = `
        <button class="btn-download-pdf" title="Download as PDF">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            <span class="btn-text">Download as PDF</span>
        </button>
        <span class="pdf-hint">You can also download this response as a PDF</span>
    `;

    const btn = actionsDiv.querySelector(".btn-download-pdf");
    btn.addEventListener("click", () => downloadPdf(responseText, btn));

    body.appendChild(actionsDiv);
}

/** Generate and download a PDF from the response text */
function downloadPdf(text, btnElement) {
    // Show loading state
    let originalText = "";
    if (btnElement) {
        const textSpan = btnElement.querySelector(".btn-text");
        if (textSpan) {
            originalText = textSpan.innerText;
            textSpan.innerText = "Downloading...";
            btnElement.disabled = true;
            btnElement.style.opacity = "0.7";
        }
    }

    // Use setTimeout to allow the browser to render the "Downloading..." state
    setTimeout(() => {
        try {
            if (!window.jspdf || !window.jspdf.jsPDF) {
                throw new Error("jsPDF library not loaded");
            }
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF({
                orientation: "portrait",
                unit: "mm",
                format: "a4",
            });

            // ── Header ──
            doc.setFillColor(10, 15, 26);
            doc.rect(0, 0, 210, 32, "F");

            doc.setFont("helvetica", "bold");
            doc.setFontSize(16);
            doc.setTextColor(56, 189, 248);
            doc.text("Healthcare AI Assistant", 15, 18);

            doc.setFont("helvetica", "normal");
            doc.setFontSize(8);
            doc.setTextColor(148, 163, 184);
            doc.text(`Generated: ${new Date().toLocaleString()}`, 15, 26);

            // ── Body ──
            doc.setFont("helvetica", "normal");
            doc.setFontSize(11);
            doc.setTextColor(30, 30, 30);

            // Clean up markdown for PDF (strip basic markdown symbols)
            const safeText = typeof text === "string" && text.trim()
                ? text
                : "No response content available.";

            const cleanText = safeText
                .replace(/#{1,6}\s?/g, "")
                .replace(/\*\*(.*?)\*\*/g, "$1")
                .replace(/\*(.*?)\*/g, "$1")
                .replace(/`(.*?)`/g, "$1")
                .replace(/---/g, "")
                .trim();

            const lines = doc.splitTextToSize(cleanText, 180);
            let y = 42;
            const pageHeight = 280;

            for (const line of lines) {
                if (y > pageHeight) {
                    doc.addPage();
                    y = 20;
                }
                doc.text(line, 15, y);
                y += 6;
            }

            // ── Footer on last page ──
            const totalPages = doc.internal.getNumberOfPages();
            for (let i = 1; i <= totalPages; i++) {
                doc.setPage(i);
                doc.setFontSize(7);
                doc.setTextColor(148, 163, 184);
                doc.text(
                    "⚠ This information is for educational purposes only. Consult a healthcare professional.",
                    105,
                    290,
                    { align: "center" }
                );
                doc.text(`Page ${i} of ${totalPages}`, 195, 290, { align: "right" });
            }

            doc.save("healthcare-response.pdf");
        } catch (error) {
            console.error("PDF Generation Error:", error);
            alert("Sorry, there was an error generating the PDF.");
        } finally {
            // Restore button state
            if (btnElement) {
                const textSpan = btnElement.querySelector(".btn-text");
                if (textSpan && originalText) {
                    textSpan.innerText = originalText;
                }
                btnElement.disabled = false;
                btnElement.style.opacity = "1";
            }
        }
    }, 50);
}

// ================================================================
// UI HELPERS
// ================================================================

/** Update the active domain pill and body theme */
function updateActiveDomain(domain, saveToDb = true) {
    state.currentDomain = domain;
    document.body.dataset.theme = domain;

    domainPills.forEach((pill) => {
        if (pill.dataset.domain === domain) {
            pill.classList.add("active");
        } else {
            pill.classList.remove("active");
        }
    });

    const welcomeBadgeIcon = $(".welcome-badge-icon");
    const welcomeBadgeText = $(".welcome-badge-text");
    if (welcomeBadgeIcon && welcomeBadgeText) {
        const domainMap = {
            medical: { icon: "🏥", name: "Medical Specialist" },
            nutrition: { icon: "🥗", name: "Nutrition Specialist" },
            therapy: { icon: "🧠", name: "Therapy Specialist" },
            teeth: { icon: "🦷", name: "Dental Specialist" },
            hair: { icon: "💇", name: "Hair Specialist" }
        };
        const info = domainMap[domain] || domainMap.medical;
        welcomeBadgeIcon.textContent = info.icon;
        welcomeBadgeText.textContent = info.name;
    }

    if (saveToDb && state.currentChatId) {
        updateChatDomain(state.currentChatId, domain);
    }
}

/** Update the domain of an existing chat in the database */
async function updateChatDomain(chatId, domain) {
    try {
        await fetch(`/api/chats/${chatId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ domain }),
        });
        // Reload list to update domain labels/classes
        const res = await fetch("/api/chats");
        const chats = await res.json();
        renderChatList(chats);
    } catch (err) {
        console.error("Failed to update chat domain:", err);
    }
}

function showWelcomeScreen() {
    welcomeScreen.style.display = "flex";
    chatArea.classList.remove("visible");
}

function showChatArea() {
    welcomeScreen.style.display = "none";
    chatArea.classList.add("visible");
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

/** Escape HTML to prevent XSS */
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
