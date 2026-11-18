
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.database import get_db, SessionLocal
from backend.models import User, UserSession, Conversation, Message
from backend.leo_api import app, lifespan


def run_all_tests():
    print("\n" + "="*50)
    print("STARTING LEO RESTORATION COMPREHENSIVE TESTS")
    print("="*50 + "\n")

    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Use TestClient with lifespan context
    with TestClient(app) as client:
        db: Session = SessionLocal()

        # ----------------------------------------------------
        # 1. SYSTEM ENDPOINTS
        # ----------------------------------------------------
        def test_home():
            res = client.get("/")
            assert res.status_code == 200
            data = res.json()
            assert data["application"] == "LEO AI"
            assert data["status"] == "running"
            assert data["intelligence_engine"] == "ollama"

        test("Home endpoint (/)", test_home)

        def test_health():
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

        test("Health check endpoint (/health)", test_health)

        # ----------------------------------------------------
        # 2. AUTHENTICATION & SECURITY TESTS
        # ----------------------------------------------------
        test_uid = uuid.uuid4().hex[:8]
        test_email = f"test_{test_uid}@example.com"
        test_password = "SecurePassword123!"
        test_name = f"Test User {test_uid}"

        user2_uid = uuid.uuid4().hex[:8]
        user2_email = f"other_{user2_uid}@example.com"
        user2_name = f"Other User {user2_uid}"

        def test_unauthenticated_chat_rejected():
            res = client.post("/chat", json={"message": "hello"})
            assert res.status_code == 401
            assert "log in" in res.json().get("detail", "").lower()

        test("Access /chat without login rejected with 401", test_unauthenticated_chat_rejected)

        def test_unauthenticated_conversations_rejected():
            res = client.get("/conversations")
            assert res.status_code == 401

        test("Access /conversations without login rejected with 401", test_unauthenticated_conversations_rejected)

        def test_signup():
            res = client.post("/auth/register", json={
                "full_name": test_name,
                "email": test_email,
                "password": test_password
            })
            assert res.status_code == 201
            data = res.json()
            assert data["email"] == test_email
            assert data["full_name"] == test_name

            # Also register second user for cross-user tests
            res2 = client.post("/auth/register", json={
                "full_name": user2_name,
                "email": user2_email,
                "password": test_password
            })
            assert res2.status_code == 201

        test("User signup (/auth/register)", test_signup)

        def test_duplicate_signup_rejected():
            res = client.post("/auth/register", json={
                "full_name": test_name,
                "email": test_email,
                "password": test_password
            })
            assert res.status_code == 409

        test("Duplicate email signup rejected with 409", test_duplicate_signup_rejected)

        def test_login_invalid_password():
            res = client.post("/auth/login", json={
                "email": test_email,
                "password": "WrongPassword!"
            })
            assert res.status_code == 401

        test("Login with incorrect password rejected with 401", test_login_invalid_password)

        def test_login_success():
            res = client.post("/auth/login", json={
                "email": test_email,
                "password": test_password
            })
            assert res.status_code == 200
            assert "leo_session" in client.cookies
            data = res.json()
            assert data["user"]["email"] == test_email

        test("Login with valid password (/auth/login)", test_login_success)

        def test_auth_me():
            res = client.get("/auth/me")
            assert res.status_code == 200
            data = res.json()
            assert data["authenticated"] is True
            assert data["user"]["email"] == test_email

        test("Verify session endpoint (/auth/me)", test_auth_me)

        # ----------------------------------------------------
        # 3. DATABASE SECURITY & INTEGRITY TESTS
        # ----------------------------------------------------
        def test_password_and_token_not_plaintext():
            user = db.query(User).filter(User.email == test_email).first()
            assert user is not None
            # Password must NOT be plaintext and must start with Argon2 header
            assert user.password_hash != test_password
            assert user.password_hash.startswith("$argon2")

            # Session token in db must be SHA-256 hash (64 hex characters)
            session_token = client.cookies.get("leo_session")
            assert session_token is not None
            session_row = db.query(UserSession).filter(UserSession.user_id == user.id).first()
            assert session_row is not None
            assert session_row.token_hash != session_token
            assert len(session_row.token_hash) == 64

        test("Passwords (Argon2) and sessions (SHA-256) not stored in plaintext", test_password_and_token_not_plaintext)

        # ----------------------------------------------------
        # 4. CHAT & QUICK RESPONSE TESTS
        # ----------------------------------------------------
        conv_id = None

        def test_quick_greeting():
            nonlocal conv_id
            res = client.post("/chat", json={
                "message": "hello",
                "search_enabled": False
            })
            assert res.status_code == 200
            data = res.json()
            assert data["assistant"] == "LEO"
            assert "Hello!" in data["response"]
            assert data["conversation_id"] is not None
            conv_id = data["conversation_id"]

            # Verify saved in MySQL
            db.rollback()
            msg_count = db.query(Message).filter(Message.conversation_id == int(conv_id)).count()
            assert msg_count == 2  # user + assistant

        test("Quick greeting response & MySQL persistence", test_quick_greeting)

        def test_ollama_ai_response():
            nonlocal conv_id
            res = client.post("/chat", json={
                "message": "What is 2 + 2? Answer in one short word.",
                "conversation_id": conv_id,
                "search_enabled": False
            })
            assert res.status_code == 200
            data = res.json()
            assert data["assistant"] == "LEO"
            assert len(data["response"]) > 0
            assert "4" in data["response"] or "four" in data["response"].lower()

            # Verify saved in MySQL
            db.rollback()
            msg_count = db.query(Message).filter(Message.conversation_id == int(conv_id)).count()
            assert msg_count == 4  # 2 exchanges

        test("Ollama AI generation with llama3.2:3b", test_ollama_ai_response)

        def test_get_conversation_history():
            res = client.get(f"/conversations/{conv_id}")
            assert res.status_code == 200
            data = res.json()
            assert data["id"] == str(conv_id)
            assert len(data["messages"]) == 4
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"

        test("Retrieve conversation history from MySQL (/conversations/{id})", test_get_conversation_history)

        def test_list_conversations():
            res = client.get("/conversations")
            assert res.status_code == 200
            data = res.json()
            assert len(data) >= 1
            ids = [c["id"] for c in data]
            assert str(conv_id) in ids

        test("List conversations for user (/conversations)", test_list_conversations)

        # ----------------------------------------------------
        # 5. CROSS-USER PRIVACY & ACCESS CONTROL TESTS
        # ----------------------------------------------------
        def test_cross_user_isolation():
            # Login as second user
            client2 = TestClient(app)
            login_res = client2.post("/auth/login", json={
                "email": user2_email,
                "password": test_password
            })
            assert login_res.status_code == 200

            # Attempt to access User 1's conversation
            res = client2.get(f"/conversations/{conv_id}")
            assert res.status_code == 404

            # Attempt to chat in User 1's conversation
            chat_res = client2.post("/chat", json={
                "message": "intruding",
                "conversation_id": conv_id
            })
            assert chat_res.status_code == 404

            # Attempt to delete User 1's conversation
            del_res = client2.delete(f"/conversations/{conv_id}")
            assert del_res.status_code == 404

        test("Cross-user privacy enforcement (User 2 blocked from User 1's chats)", test_cross_user_isolation)

        # ----------------------------------------------------
        # 6. WEB SEARCH & RESEARCH TEST
        # ----------------------------------------------------
        def test_web_research():
            res = client.post("/chat", json={
                "message": "What is Python programming language?",
                "search_enabled": True,
                "max_new_tokens": 100
            })
            assert res.status_code == 200
            data = res.json()
            assert data["assistant"] == "LEO"
            assert len(data["response"]) > 0
            assert isinstance(data["sources"], list)
            if data["sources"]:
                assert "url" in data["sources"][0]
                assert "title" in data["sources"][0]

        test("Web research with Tavily search and source links", test_web_research)

        # ----------------------------------------------------
        # 7. LOGOUT TEST
        # ----------------------------------------------------
        def test_logout():
            res = client.post("/auth/logout")
            assert res.status_code == 200

            # After logout, accessing protected endpoints must fail
            me_res = client.get("/auth/me")
            assert me_res.status_code == 401

            chat_res = client.post("/chat", json={"message": "hello"})
            assert chat_res.status_code == 401

        test("Logout (/auth/logout) and session invalidation", test_logout)

        db.close()

    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("="*50 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
