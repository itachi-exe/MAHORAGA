#!/usr/bin/env python3
"""
Builds MAHORAGA_installer.py — a single encrypted self-installing launcher.
Run this from inside the MAHORAGA folder: python3 build_installer.py

Password is prompted at build time — never stored in source code.
"""
import zipfile, io, zlib, hashlib, hmac, struct, base64, os, getpass, sys

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT     = os.path.join(os.path.dirname(SOURCE_DIR), "MAHORAGA_installer.py")

FILES = [
    "server.py",
    "core_trading_system.py",
    "MAHORAGA_dashboard.html",
    "MAHORAGA_model.pkl",
    "MAHORAGA_scaler.pkl",
    "MAHORAGA_training_data.json",
    ".env",
]

REQUIREMENTS = [
    "pybit>=5.0.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "python-dotenv>=1.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.2.0",
    "ta>=0.10.0",
    "joblib>=1.2.0",
    "anthropic>=0.20.0",
    "pydantic>=2.0.0",
    "websockets>=11.0",
    "httpx>=0.24.0",
]

def encrypt(data: bytes, password: str) -> bytes:
    salt     = os.urandom(32)
    key      = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)
    checksum = hashlib.sha256(data).digest()[:4]
    payload  = checksum + zlib.compress(data, 9)
    ks, ctr  = b"", 0
    while len(ks) < len(payload):
        ks  += hmac.new(key, struct.pack(">Q", ctr), hashlib.sha256).digest()
        ctr += 1
    return salt + bytes(a ^ b for a, b in zip(payload, ks[:len(payload)]))

# ── Prompt for password at build time ────────────────────────────
print("\n  MAHORAGA Installer Builder")
print("  ─────────────────────────")
while True:
    pw1 = getpass.getpass("  Set installer password: ")
    if len(pw1) < 8:
        print("  ✗ Password must be at least 8 characters."); continue
    pw2 = getpass.getpass("  Confirm password:       ")
    if pw1 != pw2:
        print("  ✗ Passwords do not match."); continue
    PASSWORD = pw1
    break

print()

# Bundle
print("  Bundling files...")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in FILES:
        fpath = os.path.join(SOURCE_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"    + {fname}  ({os.path.getsize(fpath):,} bytes)")
        else:
            print(f"    ! SKIPPED (not found): {fname}")

print(f"  Zip: {len(buf.getvalue()):,} bytes")
print("  Encrypting (200k PBKDF2 rounds — takes ~10s)...")
enc = encrypt(buf.getvalue(), PASSWORD)
b64 = base64.b64encode(enc).decode()
print(f"  Payload: {len(b64):,} chars")

# ── Write installer ───────────────────────────────────────────────
REQS_REPR = repr(REQUIREMENTS)

installer = f'''#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════╗
# ║       MAHORAGA AI TRADING BOT — ENCRYPTED INSTALLER     ║
# ╚══════════════════════════════════════════════════════════╝
# Requires Python 3.9+  |  Run: python3 MAHORAGA_installer.py

import sys, os, subprocess, hashlib, hmac, struct, zlib, base64
import zipfile, io, time, getpass, threading

PAYLOAD_B64  = """{b64}"""
REQUIREMENTS = {REQS_REPR}
INSTALL_DIR  = os.path.join(os.getcwd(), "MAHORAGA")

R="\\033[0;31m"; G="\\033[0;32m"; Y="\\033[0;33m"
C="\\033[0;36m"; W="\\033[1;37m"; D="\\033[2m"; X="\\033[0m"

def banner():
    print(C+"")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                                                              ║")
    print("  ║  "+W+"███╗   ███╗ █████╗ ██╗  ██╗ ██████╗ ██████╗  █████╗  "+C+"      ║")
    print("  ║  "+W+"████╗ ████║██╔══██╗██║  ██║██╔═══██╗██╔══██╗██╔══██╗ "+C+"      ║")
    print("  ║  "+W+"██╔████╔██║███████║███████║██║   ██║██████╔╝███████║ "+C+"       ║")
    print("  ║  "+W+"██║╚██╔╝██║██╔══██║██╔══██║██║   ██║██╔══██╗██╔══██║ "+C+"      ║")
    print("  ║  "+W+"██║ ╚═╝ ██║██║  ██║██║  ██║╚██████╔╝██║  ██║██║  ██║ "+C+"      ║")
    print("  ║  "+W+"╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝"+C+"      ║")
    print("  ║                                                              ║")
    print("  ║       "+D+"AI TRADING BOT  ·  ENCRYPTED SELF-INSTALLER"+C+"          ║")
    print("  ╚══════════════════════════════════════════════════════════════╝"+X)
    print()

def ok(m):   print(f"  "+G+"[✓]"+X+f" {{m}}")
def info(m): print(f"  "+C+"[→]"+X+f" {{m}}")
def err(m):  print(f"  "+R+"[✗]"+X+f" {{m}}")
def warn(m): print(f"  "+Y+"[!]"+X+f" {{m}}")

def spin(msg, done):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not done.is_set():
        print(f"  "+C+frames[i%len(frames)]+X+f" {{msg}}...", end="\\r", flush=True)
        time.sleep(0.08); i += 1
    print(" "*(len(msg)+12), end="\\r")

def decrypt(data: bytes, pw: str):
    try:
        salt, ct = data[:32], data[32:]
        key = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000, dklen=32)
        ks, ctr = b"", 0
        while len(ks) < len(ct):
            ks += hmac.new(key, struct.pack(">Q", ctr), hashlib.sha256).digest()
            ctr += 1
        raw = bytes(a ^ b for a, b in zip(ct, ks[:len(ct)]))
        checksum, compressed = raw[:4], raw[4:]
        dec = zlib.decompress(compressed)
        if hashlib.sha256(dec).digest()[:4] != checksum:
            return None
        return dec
    except Exception:
        return None

def install_deps(pip_bin):
    total = len(REQUIREMENTS)
    for i, pkg in enumerate(REQUIREMENTS, 1):
        name  = pkg.split(">=")[0].split("==")[0]
        label = f"[{{i:>2}}/{{total}}] {{name:<22}}"
        done  = threading.Event()
        t     = threading.Thread(target=spin, args=(label, done), daemon=True)
        t.start()
        r = subprocess.run([pip_bin,"install",pkg,"-q","--disable-pip-version-check"],
                           capture_output=True)
        done.set(); t.join()
        status = G+"[✓]"+X if r.returncode == 0 else Y+"[!]"+X
        print(f"  {{status}} {{label}}")

def setup_venv(d):
    venv = os.path.join(d, ".venv")
    py   = os.path.join(venv, "bin", "python3")
    pip  = os.path.join(venv, "bin", "pip")
    if not os.path.exists(py):
        info("Creating virtual environment...")
        subprocess.run([sys.executable,"-m","venv",venv], check=True, capture_output=True)
        ok("Virtual environment created")
    else:
        ok("Virtual environment ready")
    return py, pip

def main():
    banner()

    if sys.version_info < (3, 9):
        err(f"Python 3.9+ required. You have {{sys.version.split()[0]}}"); sys.exit(1)
    ok(f"Python {{sys.version.split()[0]}}")
    print()

    zip_data = None
    for attempt in range(1, 4):
        try:
            pw = getpass.getpass("  "+W+"Enter access password: "+X)
        except (EOFError, KeyboardInterrupt):
            print(); err("Cancelled."); sys.exit(1)

        ev = threading.Event()
        t  = threading.Thread(target=spin, args=("Verifying", ev), daemon=True)
        t.start()
        result = decrypt(base64.b64decode(PAYLOAD_B64), pw)
        ev.set(); t.join()

        if result is None:
            print()
            print("  "+R+"╔═══════════════════════════════╗")
            print("  ║   ⛔   ACCESS DENIED           ║")
            print("  ╚═══════════════════════════════╝"+X)
            if attempt < 3:
                warn(f"{{3-attempt}} attempt(s) remaining."); print()
        else:
            print()
            print("  "+G+"╔═══════════════════════════════╗")
            print("  ║   ✅   ACCESS GRANTED          ║")
            print("  ╚═══════════════════════════════╝"+X)
            zip_data = result; break

    if zip_data is None:
        err("Too many failed attempts."); sys.exit(1)

    print()
    os.makedirs(INSTALL_DIR, exist_ok=True)
    info(f"Extracting to {{INSTALL_DIR}}/")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for name in zf.namelist():
            zf.extract(name, INSTALL_DIR)
            ok(name)
    print()

    info("Setting up Python environment...")
    venv_py, venv_pip = setup_venv(INSTALL_DIR)
    print()
    info("Installing dependencies (first run may take a few minutes)...")
    print()
    install_deps(venv_pip)
    print()
    ok("All dependencies installed")
    print()

    print("  "+C+"  ══════════════════════════════════════════════"+X)
    print("  "+G+"  ✅  MAHORAGA is starting..."+X)
    print("  "+W+"     Dashboard → http://localhost:8501"+X)
    print("  "+C+"  ══════════════════════════════════════════════"+X)
    print()

    os.chdir(INSTALL_DIR)
    os.execv(venv_py, [venv_py, os.path.join(INSTALL_DIR, "server.py")])

if __name__ == "__main__":
    main()
'''

with open(OUTPUT, "w") as f:
    f.write(installer)

sz = os.path.getsize(OUTPUT) / 1024
print(f"\n  ✓ Done  →  {OUTPUT}")
print(f"  Size     {sz:.0f} KB")
print(f"  Password is NOT stored in source — keep it safe.\n")
