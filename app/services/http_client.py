import logging
import random
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class HttpClient:
    def __init__(self, timeout_s: float = 6.0, max_retries: int = 3, backoff_base_s: float = 0.4):
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> requests.Response:
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                )
                # Retry on 429/5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable_status={resp.status_code}", response=resp)
                return resp
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = (self.backoff_base_s * (2 ** (attempt - 1))) * (1.0 + random.random() * 0.2)
                logger.warning(
                    "HTTP retry attempt=%s/%s method=%s url=%s error=%s backoff_s=%.2f",
                    attempt,
                    self.max_retries,
                    method,
                    url,
                    str(e)[:240],
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise last_err or RuntimeError("http_request_failed")

    def get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        resp = self.request("GET", url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def post_json(
        self, url: str, *, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        resp = self.request("POST", url, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def get_text(self, url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
        resp = self.request("GET", url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.text


http_client = HttpClient(timeout_s=6.0, max_retries=3)

