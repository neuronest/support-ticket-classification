from typing import Optional
import requests
import json


def post_classify_ticket(
    message: str, url: str = "http://127.0.0.1", port: Optional[int] = None
):
    post_url = f"{url}"
    post_url += f":{port}" if port is not None else ""
    post_url += "/ticket_support_classification"
    body = {"message": message}
    headers = {"Content-Type": "application/json"}
    response = requests.request(
        "POST", post_url, headers=headers, data=json.dumps(body)
    )
    return json.loads(response.text.encode("utf8"))
