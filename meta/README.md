# Local data directory

Place the ELMA annotation data under this directory. The default layout is:

```text
meta/
├── test_anaphora_per_category4_full_episodes.jsonl
└── test_videos/
    ├── <video>.mp4
    └── ...
```

Run `python app.py` from the repository root to use this layout. The application
resolves these defaults relative to `app.py`, so it also works when launched
from another working directory.

Exports are kept outside the dataset directory under `../annotation_output/`.
The application can still resume a legacy `*_annotated.jsonl` file in this
directory, but all new default exports use the versioned output directory.

Actual JSONL and video files are intentionally ignored by Git because they may
be large and may have separate distribution or privacy requirements.
