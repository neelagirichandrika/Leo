
const API_URL = "http://127.0.0.1:8000";

const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatMessages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");
const newChatButton = document.getElementById("new-chat");
const searchToggle = document.getElementById("search-toggle");
const conversationList = document.getElementById("conversation-list");
const userDisplay = document.getElementById("user-display");
const logoutBtn = document.getElementById("logout-btn");
const statusText = document.getElementById("status-text");

let isWaiting = false;
let currentConversationId = null;
let currentUser = null;


// ==========================================
// AUTHENTICATION CHECK & INITIALIZATION
// ==========================================

async function checkSession() {
    try {
        const response = await fetch(`${API_URL}/auth/me`, {
            method: "GET",
            credentials: "include"
        });

        if (!response.ok) {
            window.location.href = "login.html";
            return;
        }

        const data = await response.json();
        currentUser = data.user;

        if (userDisplay && currentUser) {
            userDisplay.textContent = currentUser.full_name || currentUser.email;
            userDisplay.title = currentUser.email;
        }

        // Initialize status and conversations
        loadAppStatus();
        loadConversations();

    } catch (error) {
        console.error("Session check error:", error);
        window.location.href = "login.html";
    }
}


// ==========================================
// LOAD ENGINE STATUS
// ==========================================

async function loadAppStatus() {
    try {
        const response = await fetch(`${API_URL}/`, {
            method: "GET"
        });

        if (response.ok) {
            const data = await response.json();
            if (statusText) {
                const engineName = data.model || data.intelligence_engine || "LEO AI";
                statusText.textContent = engineName;
            }
        }
    } catch (error) {
        console.warn("Status fetch failed:", error);
    }
}


// ==========================================
// LOGOUT
// ==========================================

if (logoutBtn) {
    logoutBtn.addEventListener("click", async function() {
        try {
            await fetch(`${API_URL}/auth/logout`, {
                method: "POST",
                credentials: "include"
            });
        } catch (error) {
            console.error("Logout request error:", error);
        } finally {
            window.location.href = "login.html";
        }
    });
}


// ==========================================
// CONVERSATION SIDEBAR MANAGEMENT
// ==========================================

async function loadConversations() {
    if (!conversationList) return;

    try {
        const response = await fetch(`${API_URL}/conversations`, {
            method: "GET",
            credentials: "include"
        });

        if (!response.ok) {
            return;
        }

        const conversations = await response.json();
        renderConversationList(conversations);

    } catch (error) {
        console.error("Error loading conversations:", error);
    }
}


function renderConversationList(conversations) {
    if (!conversationList) return;

    conversationList.innerHTML = "";

    if (!Array.isArray(conversations) || conversations.length === 0) {
        const emptyNotice = document.createElement("div");
        emptyNotice.className = "sidebar-info";
        emptyNotice.style.fontSize = "12px";
        emptyNotice.style.textTransform = "none";
        emptyNotice.textContent = "No conversations yet";
        conversationList.appendChild(emptyNotice);
        return;
    }

    conversations.forEach((conv) => {
        const item = document.createElement("div");
        item.className = "conversation-item";
        if (conv.id === currentConversationId) {
            item.classList.add("active");
        }

        const titleSpan = document.createElement("span");
        titleSpan.className = "conv-title";
        titleSpan.textContent = conv.title || "Untitled chat";
        titleSpan.title = "Double-click to rename";
        titleSpan.addEventListener("dblclick", function(event) {
            event.stopPropagation();
            renameConversationPrompt(conv.id, conv.title || "Untitled chat");
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "delete-conv-btn";
        deleteBtn.innerHTML = "×";
        deleteBtn.title = "Delete conversation";
        deleteBtn.addEventListener("click", function(event) {
            event.stopPropagation();
            deleteConversation(conv.id);
        });

        item.appendChild(titleSpan);
        item.appendChild(deleteBtn);

        item.addEventListener("click", function() {
            if (!isWaiting) {
                selectConversation(conv.id);
            }
        });

        conversationList.appendChild(item);
    });
}


async function selectConversation(convId) {
    try {
        const response = await fetch(`${API_URL}/conversations/${convId}`, {
            method: "GET",
            credentials: "include"
        });

        if (!response.ok) {
            throw new Error("Unable to load conversation.");
        }

        const data = await response.json();
        currentConversationId = data.id;

        // Clear chat view
        chatMessages.innerHTML = "";

        // Highlight active item in sidebar
        const items = document.querySelectorAll(".conversation-item");
        items.forEach((item) => {
            const titleEl = item.querySelector(".conv-title");
            if (titleEl && titleEl.textContent === data.title) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        // Render past messages
        if (Array.isArray(data.messages) && data.messages.length > 0) {
            data.messages.forEach((msg) => {
                const roleLabel = msg.role === "user" ? "You" : "LEO";
                addMessage(roleLabel, msg.content);
            });
        } else {
            renderWelcomeScreen();
        }

        loadConversations();

    } catch (error) {
        console.error("Failed to load conversation:", error);
    }
}


async function deleteConversation(convId) {
    if (!confirm("Are you sure you want to delete this conversation?")) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/conversations/${convId}`, {
            method: "DELETE",
            credentials: "include"
        });

        if (!response.ok) {
            throw new Error("Failed to delete conversation.");
        }

        if (currentConversationId === convId) {
            startNewChat();
        } else {
            loadConversations();
        }

    } catch (error) {
        console.error("Delete conversation error:", error);
    }
}


async function renameConversationPrompt(convId, currentTitle) {
    const newTitle = prompt("Enter a new title for this conversation:", currentTitle);
    if (!newTitle || newTitle.trim() === "" || newTitle.trim() === currentTitle) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/conversations/${convId}`, {
            method: "PATCH",
            headers: {
                "Content-Type": "application/json"
            },
            credentials: "include",
            body: JSON.stringify({ title: newTitle.trim() })
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Failed to rename conversation.");
        }

        loadConversations();
    } catch (error) {
        console.error("Rename conversation error:", error);
    }
}


function renderWelcomeScreen() {
    chatMessages.innerHTML = `
        <div class="welcome-screen" id="welcome-screen">
            <div class="welcome-logo">L</div>
            <h1>How can I help you?</h1>
            <p>Welcome to LEO, your personal AI assistant.</p>
        </div>
    `;
}


function startNewChat() {
    if (isWaiting) return;

    currentConversationId = null;
    renderWelcomeScreen();

    // Deselect active conversation item
    const activeItems = document.querySelectorAll(".conversation-item.active");
    activeItems.forEach((item) => item.classList.remove("active"));

    messageInput.value = "";
    messageInput.style.height = "auto";
    messageInput.focus();

    loadConversations();
}


// ==========================================
// CREATE MESSAGE BUBBLE
// ==========================================

function addMessage(role, content, sources = []) {
    const welcome = document.getElementById("welcome-screen");
    if (welcome) {
        welcome.remove();
    }

    const row = document.createElement("div");
    row.className = "message-row";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "You" ? "Y" : "L";

    const contentBox = document.createElement("div");

    const roleLabel = document.createElement("div");
    roleLabel.className = "message-role";
    roleLabel.textContent = role;

    const messageText = document.createElement("div");
    messageText.className = "message-content";
    messageText.textContent = content;

    contentBox.appendChild(roleLabel);
    contentBox.appendChild(messageText);

    // Display web sources if available
    if (role === "LEO" && Array.isArray(sources) && sources.length > 0) {
        const sourcesContainer = document.createElement("div");
        sourcesContainer.className = "message-sources";

        const sourcesHeading = document.createElement("div");
        sourcesHeading.className = "sources-heading";
        sourcesHeading.textContent = "Sources";
        sourcesContainer.appendChild(sourcesHeading);

        sources.forEach((source) => {
            if (!source.title || !source.url) return;

            try {
                const parsedUrl = new URL(source.url);
                if (!["http:", "https:"].includes(parsedUrl.protocol)) return;

                const link = document.createElement("a");
                link.className = "source-link";
                link.href = parsedUrl.href;
                link.textContent = source.title;
                link.target = "_blank";
                link.rel = "noopener noreferrer";

                sourcesContainer.appendChild(link);
            } catch (error) {
                console.warn("Invalid source URL:", source.url);
            }
        });

        if (sourcesContainer.children.length > 1) {
            contentBox.appendChild(sourcesContainer);
        }
    }

    row.appendChild(avatar);
    row.appendChild(contentBox);

    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ==========================================
// THINKING INDICATOR
// ==========================================

function showThinkingIndicator(isResearching = false) {
    removeThinkingIndicator();

    const welcome = document.getElementById("welcome-screen");
    if (welcome) {
        welcome.remove();
    }

    const row = document.createElement("div");
    row.className = "message-row typing-row";
    row.id = "leo-thinking-indicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "L";

    const contentBox = document.createElement("div");

    const roleLabel = document.createElement("div");
    roleLabel.className = "message-role";
    roleLabel.textContent = "LEO";

    const indicator = document.createElement("div");
    indicator.className = "typing-indicator";
    indicator.setAttribute("aria-live", "polite");

    const status = document.createElement("span");
    status.className = "typing-status";
    status.textContent = isResearching ? "LEO is researching..." : "LEO is thinking...";

    const dots = document.createElement("div");
    dots.className = "typing-dots";

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.className = "typing-dot";
        dots.appendChild(dot);
    }

    indicator.appendChild(status);
    indicator.appendChild(dots);

    contentBox.appendChild(roleLabel);
    contentBox.appendChild(indicator);

    row.appendChild(avatar);
    row.appendChild(contentBox);

    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function removeThinkingIndicator() {
    const indicator = document.getElementById("leo-thinking-indicator");
    if (indicator) {
        indicator.remove();
    }
}


// ==========================================
// SEND MESSAGE TO LEO
// ==========================================

async function sendMessage(message) {
    if (isWaiting) return;

    isWaiting = true;

    addMessage("You", message);
    sendButton.disabled = true;

    const searchEnabled = searchToggle ? searchToggle.checked : false;
    showThinkingIndicator(searchEnabled);

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            credentials: "include",
            body: JSON.stringify({
                message: message,
                conversation_id: currentConversationId,
                max_new_tokens: 900,
                temperature: 0.4,
                search_enabled: searchEnabled
            })
        });

        const data = await response.json();

        if (response.status === 401) {
            window.location.href = "login.html";
            return;
        }

        if (!response.ok) {
            throw new Error(data.detail || `Server error: ${response.status}`);
        }

        if (!data.response) {
            throw new Error("LEO returned an empty response.");
        }

        removeThinkingIndicator();

        // Update active conversation ID and display response
        currentConversationId = data.conversation_id;
        addMessage("LEO", data.response, data.sources || []);

        // Refresh sidebar conversation titles
        loadConversations();

    } catch (error) {
        console.error("LEO API Error:", error);
        removeThinkingIndicator();
        addMessage("LEO", "Unable to connect to LEO: " + error.message);

    } finally {
        removeThinkingIndicator();
        isWaiting = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}


// ==========================================
// EVENT LISTENERS
// ==========================================

chatForm.addEventListener("submit", async function(event) {
    event.preventDefault();

    const message = messageInput.value.trim();
    if (!message || isWaiting) return;

    messageInput.value = "";
    messageInput.style.height = "auto";

    await sendMessage(message);
});


newChatButton.addEventListener("click", function() {
    startNewChat();
});


messageInput.addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});


messageInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 140) + "px";
});


// Start session verification on load
checkSession();
