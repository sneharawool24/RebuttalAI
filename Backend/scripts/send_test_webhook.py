"""Send the harmless signed webhook fixture to a locally running backend.

Usage from Backend/: python scripts/send_test_webhook.py
Set RAZORPAY_WEBHOOK_SECRET in the local environment (or Backend/.env) first.
Optionally set REBUTTALAI_WEBHOOK_URL for a different local endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = BACKEND_DIR / "tests" / "fixtures" / "payment_dispute_created.json"
DEFAULT_URL = "http://localhost:8000/webhooks/razorpay"


def main() -> None:
    load_dotenv(BACKEND_DIR / ".env")
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")
    if not secret:
        raise SystemExit(
            "Set RAZORPAY_WEBHOOK_SECRET locally before running this helper."
        )

    raw_body = FIXTURE_PATH.read_bytes()
    signature = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    request = Request(
        os.getenv("REBUTTALAI_WEBHOOK_URL", DEFAULT_URL),
        data=raw_body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": signature,
            "x-razorpay-event-id": "evt_test_rebuttalai_001",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            print(response.read().decode("utf-8"))
    except HTTPError as error:
        print(error.read().decode("utf-8"))
        raise SystemExit(error.code) from error
    except URLError as error:
        raise SystemExit("Could not reach the local RebuttalAI backend.") from error


if __name__ == "__main__":
    main()
