import os, yaml, re
from dotenv import load_dotenv

_env_pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

def _expand_env(val: str) -> str:
    def repl(m):
        key, default = m.group(1), m.group(2)
        return os.getenv(key, default or "")
    return _env_pattern.sub(repl, val)

def load_config(path: str) -> dict:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = _expand_env(raw)
    return yaml.safe_load(raw)
