from __future__ import annotations

import argparse
import json
from pathlib import Path

from spincore.production_evidence_capture import build_selected_state_packet_from_capture


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bind an official client/rule-document capture to one exact R8 selected-state evidence packet"
    )
    ap.add_argument("--capture", type=Path, required=True, help="original captured file bytes")
    ap.add_argument("--spec", type=Path, required=True, help="explicit selected-state JSON facts")
    ap.add_argument("--out", type=Path, required=True, help="output evidence packet JSON")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    packet = build_selected_state_packet_from_capture(capture_path=args.capture, spec=spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "packet_id": packet.packet_id,
        "capture_sha256": packet.capture_sha256,
        "capture_size_bytes": packet.capture_size_bytes,
        "state": [packet.table_size, packet.buy_in_minor_units, packet.multiplier],
        "ready_for_tables": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
