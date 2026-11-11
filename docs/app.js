
const API_URL = "http://127.0.0.1:8000";

const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatMessages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");
const newChatButton = document.getElementById("new-chat");

const searchToggle = document.getElementById("search-toggle");

let isWaiting = false;


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

    // Display web sources

    if (role === "LEO" && Array.isArray(sources) && sources.length > 0) {

        const sourcesContainer = document.createElement("div");
        sourcesContainer.className = "message-sources";

        const sourcesHeading = document.createElement("div");
        sourcesHeading.className = "sources-heading";
        sourcesHeading.textContent = "Sources";

        sourcesContainer.appendChild(sourcesHeading);

        sources.forEach((source) => {

            if (!source.title || !source.url) {
                return;
            }

            try {

                const parsedUrl = new URL(source.url);

                if (!["http:", "https:"].includes(parsedUrl.protocol)) {
                    return;
                }

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

function showThinkingIndicator() {

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
    status.textContent = "LEO is thinking...";

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


// ==========================================
// REMOVE THINKING INDICATOR
// ==========================================

function removeThinkingIndicator() {

    const indicator = document.getElementById(
        "leo-thinking-indicator"
    );

    if (indicator) {
        indicator.remove();
    }
}


// ==========================================
// SEND MESSAGE TO LEO
// ==========================================

async function sendMessage(message) {

    if (isWaiting) {
        return;
    }

    isWaiting = true;

    addMessage("You", message);

    sendButton.disabled = true;

    // Show indicator immediately
    showThinkingIndicator();

    const searchEnabled = searchToggle
        ? searchToggle.checked
        : false;

    try {

        const response = await fetch(`${API_URL}/chat`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                message: message,
                max_new_tokens: 300,
                temperature: 0.4,
                search_enabled: searchEnabled

            })

        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail || `Server error: ${response.status}`
            );

        }

        if (!data.response) {

            throw new Error(
                "LEO returned an empty response."
            );

        }

        removeThinkingIndicator();

        addMessage(
            "LEO",
            data.response,
            data.sources || []
        );

    } catch (error) {

        console.error("LEO API Error:", error);

        removeThinkingIndicator();

        addMessage(
            "LEO",
            "Unable to connect to LEO: " + error.message
        );

    } finally {

        removeThinkingIndicator();

        isWaiting = false;

        sendButton.disabled = false;

        messageInput.focus();

    }
}


// ==========================================
// FORM SUBMISSION
// ==========================================

chatForm.addEventListener("submit", async function(event) {

    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message || isWaiting) {
        return;
    }

    messageInput.value = "";
    messageInput.style.height = "auto";

    await sendMessage(message);

});


// ==========================================
// NEW CHAT
// ==========================================

newChatButton.addEventListener("click", function() {

    if (isWaiting) {
        return;
    }

    chatMessages.innerHTML = `
        <div class="welcome-screen" id="welcome-screen">
            <div class="welcome-logo">L</div>
            <h1>How can I help you?</h1>
            <p>Welcome to LEO, your personal AI assistant.</p>
        </div>
    `;

    messageInput.value = "";
    messageInput.style.height = "auto";
    messageInput.focus();

});


// ==========================================
// AUTO-RESIZE INPUT
// ==========================================

messageInput.addEventListener("input", function() {

    this.style.height = "auto";

    this.style.height = Math.min(
        this.scrollHeight,
        140
    ) + "px";

});
