# Local data directory

Place the ELMA annotation data under this directory. The default layout is:

```text
meta/
├── test_anaphora_per_category4_full_episodes.jsonl
├── test_anaphora_per_category4_full_episodes_annotated.jsonl  # created/resumed by the app
└── test_videos/
    ├── <video>.mp4
    └── ...
```

Run `python app.py` from the repository root to use this layout. The application
resolves these defaults relative to `app.py`, so it also works when launched
from another working directory.

Actual JSONL and video files are intentionally ignored by Git because they may
be large and may have separate distribution or privacy requirements.
