
const API_URL = "http://127.0.0.1:8001";

const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatMessages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");
const newChatButton = document.getElementById("new-chat");

// ==========================================
// CONVERSATION MEMORY
// ==========================================

let conversationId = null;


// ==========================================
// CREATE MESSAGE BUBBLE
// ==========================================

function addMessage(role, content) {

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

    row.appendChild(avatar);
    row.appendChild(contentBox);

    chatMessages.appendChild(row);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ==========================================
// SEND MESSAGE TO LEO
// ==========================================

async function sendMessage(message) {

    addMessage("You", message);

    sendButton.disabled = true;

    try {

        const response = await fetch(`${API_URL}/chat`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                message: message,

                conversation_id: conversationId,

                max_new_tokens: 150,

                temperature: 0.4

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

        // Save the ID returned by the backend.
        conversationId = data.conversation_id;

        addMessage("LEO", data.response);

    } catch (error) {

        console.error("LEO API Error:", error);

        addMessage(
            "LEO",
            "Unable to connect to LEO: " + error.message
        );

    } finally {

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

    if (!message) {
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

    // Reset conversation memory.
    conversationId = null;

    // Restore welcome screen.
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
