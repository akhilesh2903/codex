import os
import ssl
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://akhibhat777_db_user:Kt9RtHXlkfGfJ7ls@cluster0.pqxhyok.mongodb.net/dr_screening?appName=Cluster0&retryWrites=true&w=majority"
)

_client = None


import certifi

def _make_client():
    """Create MongoClient with certifi TLS options that works on Windows / Python 3.11+."""
    try:
        c = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
            tls=True,
            tlsCAFile=certifi.where(),
        )
        c.admin.command("ping")
        print("[DB]    ✅ MongoDB connected (certifi TLS).")
        return c
    except Exception as e:
        print(f"[DB]    ❌ Connection check failed: {e}")
        try:
            # Fallback for Windows without cert verification
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            c = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                tls=True,
                tlsAllowInvalidCertificates=True,
            )
            c.admin.command("ping")
            print("[DB]    ✅ MongoDB connected (fallback without cert check).")
            return c
        except Exception as e2:
            print(f"[DB]    ❌ All connection attempts failed: {e2}")
            return None


try:
    _client = _make_client()
except Exception as e:
    print(f"[ERROR] Failed to create MongoDB client: {e}")
    _client = None

client = _client
db = client["dr_screening"] if client else None


def verify_connection():
    """Ping MongoDB to confirm the connection is alive. Returns True/False."""
    global client, db
    if client is None:
        # Try to reconnect
        client = _make_client()
        db = client["dr_screening"] if client else None
    if client is None:
        return False
    try:
        client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"[ERROR] MongoDB ping failed: {e}")
        return False

