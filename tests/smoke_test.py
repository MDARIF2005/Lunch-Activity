import requests
from datetime import datetime, timedelta

BASE = "http://127.0.0.1:5000"

def assert_status(response, expect_code=200):
    assert response.status_code == expect_code, f"Unexpected {response.status_code} for {response.url}"
    return response

def test_index():
    r = requests.get(f"{BASE}/")
    assert_status(r)

def post_selection(launch_type, value, trigger):
    files = {}
    data = {
        "launch_type": launch_type,
        "trigger_type": trigger,
    }
    if launch_type == "website":
        data["website_url"] = value
    elif launch_type == "image":
        data["image_path"] = value
    else:
        data["file_path"] = value
    r = requests.post(f"{BASE}/store_selection", data=data, files=files, allow_redirects=False)
    assert r.status_code in (302, 303), f"Expected redirect, got {r.status_code}"
    return r.headers.get("Location")

def test_button():
    loc = post_selection("website", "https://example.com", "button")
    r = requests.get(f"{BASE}{loc}")
    assert_status(r)

def test_hand():
    loc = post_selection("image", "/path/to/fake.jpg", "hand")
    r = requests.get(f"{BASE}{loc}")  # /hand
    assert_status(r)
    # simulate trigger
    r2 = requests.post(f"{BASE}/hand_trigger", allow_redirects=False)
    assert r2.status_code in (302, 303)

def test_voice():
    loc = post_selection("website", "https://example.com", "voice")
    r = requests.get(f"{BASE}{loc}")  # /voice
    assert_status(r)

def test_timer():
    loc = post_selection("website", "https://example.com", "timer")
    r = requests.get(f"{BASE}{loc}")
    assert_status(r)
    launch_time = (datetime.now() + timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M')
    r2 = requests.post(f"{BASE}/schedule_launch", data={"launch_time": launch_time})
    assert_status(r2)

def test_qr():
    loc = post_selection("website", "https://example.com", "qr")
    r = requests.get(f"{BASE}{loc}")
    assert_status(r)

def test_remote():
    loc = post_selection("website", "https://example.com", "remote")
    r = requests.get(f"{BASE}{loc}")
    assert_status(r)
    r2 = requests.post(f"{BASE}/remote_launch", allow_redirects=False)
    assert r2.status_code in (302, 303)

def test_proximity():
    loc = post_selection("website", "https://example.com", "proximity")
    r = requests.get(f"{BASE}{loc}")
    assert_status(r)

if __name__ == "__main__":
    test_index()
    test_button()
    test_hand()
    test_voice()
    test_timer()
    test_qr()
    test_remote()
    test_proximity()
    print("OK: all smoke tests passed (routes reachable and basic flows respond)")


