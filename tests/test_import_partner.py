"""Personal-data check for partner sentences.

    python tests/test_import_partner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import import_partner as ip  # noqa: E402

DROP = ["Schreib an max.muster@firma.de", "Ruf 0171 2345678 an", "Ruf +49 30 1234567 an",
        "IBAN DE89 3704 0044 0532 0130 00", "schau auf www.beispiel.de", "https://x.org",
        "Treffen Hauptstraße 12", "Lindenweg 3a"]
KEEP = ["Termin morgen um 14 Uhr", "Rechnung 4711 prüfen", "am 14.11. um 9:30",
        "Paletten im Lager B bis Freitag", "Ring mich an", "Termin mit Herrn Weg"]

if __name__ == "__main__":
    for text in DROP:
        assert ip.personal_hits(text), text
    for text in KEEP:
        assert not ip.personal_hits(text), (text, ip.personal_hits(text))
    print("ok")
