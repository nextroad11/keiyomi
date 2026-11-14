"""data ブランチへの保存。中身はすべて暗号化してから置く。

- enc/  : 合言葉（APP_PASS）で暗号化。アプリと GitHub Actions が読む。
- dev/  : keys/dev_pub.txt の公開鍵で封をした控え。開発中に Claude が中身を確かめるためのもの。
          秘密鍵は Claude の作業環境にだけあり、リポジトリには置かない。不要になったら dev/ ごと消す。
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from crypt import _pass, data_key, decrypt_bytes, encrypt_bytes, seal

ROOT = Path(__file__).resolve().parent.parent
DEV_PUB = ROOT / "keys" / "dev_pub.txt"


class Store:
    def __init__(self, data: Path):
        self.data = Path(data)
        self.key = data_key(_pass())  # APP_PASS が無ければここで止まる
        self.dev = DEV_PUB.read_text().strip() if DEV_PUB.exists() and os.environ.get("KEIYOMI_DEV", "1") == "1" else None

    def read(self, rel: str, default=None):
        f = self.data / "enc" / rel
        if not f.exists():
            return default
        raw = decrypt_bytes(self.key, json.loads(f.read_text(encoding="utf-8")))
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw)

    def write(self, rel: str, obj, gz: bool = False) -> None:
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if gz:
            raw = gzip.compress(raw, 9)
        f = self.data / "enc" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(encrypt_bytes(self.key, raw)), encoding="utf-8")
        if self.dev:
            g = self.data / "dev" / rel
            g.parent.mkdir(parents=True, exist_ok=True)
            g.write_text(json.dumps(seal(self.dev, raw)), encoding="utf-8")
