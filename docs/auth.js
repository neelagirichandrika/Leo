
const API_URL = "http://127.0.0.1:8000";

const signupForm = document.getElementById("signup-form");
const loginForm = document.getElementById("login-form");

// SIGNUP
if (signupForm) {
    signupForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const fullName = document.getElementById("full-name").value.trim();
        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value;
        const confirmPassword = document.getElementById("confirm-password").value;
        const authMessage = document.getElementById("auth-message");

        if (password !== confirmPassword) {
            authMessage.textContent = "Passwords do not match.";
            return;
        }

        authMessage.textContent = "Creating your account...";

        try {
            const response = await fetch(`${API_URL}/auth/register`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    full_name: fullName,
                    email: email,
                    password: password
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Registration failed.");
            }

            authMessage.textContent = "Account created successfully! Redirecting...";
            window.location.href = "login.html";

        } catch (error) {
            authMessage.textContent = error.message;
        }
    });
}

// LOGIN — we will implement this in the next step.
if (loginForm) {
    loginForm.addEventListener("submit", function (event) {
        event.preventDefault();
        document.getElementById("auth-message").textContent =
            "Login API is not implemented yet.";
    });
}
