from typing import Any

PREVIEW = "https://cdnt-preview.dzcdn.net/api/1/1/a/b/c.mp3?hdnea=exp=1234~hmac=abc"


def track_json(
    track_id: int = 3135556,
    title: str = "Creep",
    artist: str = "Radiohead",
    bpm: float = 91.9,
    preview: str = PREVIEW,
) -> dict[str, Any]:
    return {
        "id": track_id,
        "title": title,
        "title_short": title,
        "duration": 238,
        "rank": 900000,
        "isrc": "GBAYE9200070",
        "bpm": bpm,
        "preview": preview,
        "artist": {"id": 399, "name": artist, "picture": "x", "preview": preview},
        "album": {
            "id": 302127,
            "title": "Pablo Honey",
            "cover_big": "https://cdn/cover_big.jpg",
        },
    }


def search_json(*tracks: dict[str, Any]) -> dict[str, Any]:
    return {"data": list(tracks), "total": len(tracks)}


def error_json(type_: str, message: str, code: int) -> dict[str, Any]:
    return {"error": {"type": type_, "message": message, "code": code}}


QUOTA_ERROR = error_json("Exception", "Quota limit exceeded", 4)
NOT_FOUND_ERROR = error_json("DataException", "no data", 800)
