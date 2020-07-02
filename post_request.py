import requests
import json


def post_classify_ticket(message, url="http://127.0.0.1", port=None):
    if port is None:
        port = ""
    post_url = f"{url}:{port}/ticket_support_classification"
    body = {"message": message}
    headers = {"Content-Type": "application/json"}
    response = requests.request(
        "POST", post_url, headers=headers, data=json.dumps(body)
    )
    return json.loads(response.text.encode("utf8"))
