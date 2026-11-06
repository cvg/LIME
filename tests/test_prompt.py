from pathlib import Path

from lime_camera.prompt import build_messages


def test_user_turn_is_image_then_bare_intent():
    messages = build_messages("room.jpg", "Find my bike")
    assert messages == [{"role": "user", "content": [
        {"type": "image", "image": "room.jpg"},
        {"type": "text", "text": "Find my bike"},
    ]}]


def test_path_images_become_strings():
    assert build_messages(Path("a/room.jpg"), "x")[0]["content"][0]["image"] == "a/room.jpg"
