# Anaphoric Annotation Tool

A lightweight Flask web interface for reviewing motion episodes alongside
their videos and correcting event-anaphora annotations.

## Features

- Browse all episodes in a paginated sidebar with 100 episodes per page.
- Restrict each annotator to a shareable, inclusive episode assignment range.
- Search and filter episodes and captions across the full dataset.
- Play the video associated with each motion segment.
- Edit and save `target_caption`, `event_anaphora`, and `keep_body_parts`.
- Mark one or more action-switch times directly from the video player.
- Require an explicit `no_action_switch` decision when a video has no switch.
- Save annotations as JSONL and resume from an existing output file.

## Installation

Python 3.8 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

### Default ELMA layout

Put the ELMA JSONL and videos in the repository-local `meta/` directory:

```text
anaphoric_annotation/
├── app.py
└── meta/
    ├── test_anaphora_per_category4_full_episodes.jsonl
    └── test_videos/
        └── *.mp4
```

Then start the default ELMA annotator without absolute data paths:

```bash
python app.py --host 127.0.0.1 --port 8888
```

The defaults are resolved relative to the directory containing `app.py`, not
the current shell directory. This means the same command and repository layout
remain portable after cloning or moving the project. See [`meta/README.md`](meta/README.md)
for the complete local data layout.

### Assigning episode ranges

Use the **Episode assignment range** controls in the sidebar to restrict an
annotator to a subset of the dataset. Positions are 1-based and inclusive, so
`200-300` contains 101 episodes; use `201-300` for exactly 100 episodes.

Applying a range updates the URL, for example:

```text
http://127.0.0.1:8888/?range_start=201&range_end=300
```

The sidebar, search, filters, pagination, and episode Previous/Next navigation
stay inside the selected range. **Save Assigned Range** exports only the
episodes in that inclusive range, in their original entry order. The range is
also included in the filename, such as `_episodes_201-300.jsonl`, so different
assignments do not overwrite each other.

### Custom flat JSONL and video directory

```bash
python app.py /path/to/input.jsonl /path/to/videos \
  --host 127.0.0.1 \
  --port 8888
```

Each JSONL record should normally contain:

- `episode_id`
- `target_segment_id`
- `target_video_name`
- `target_caption`

`target_caption` is editable and must remain non-empty. The other editable
annotation fields may already be present or will use empty defaults.

### Annotation workflow

For each segment:

1. Decide whether the action contains a state that must be maintained while
   another action happens. If so, enable `event_anaphora`.
2. Select the maintained state's body parts in `keep_body_parts`.
3. Check that `target_caption` explicitly describes the maintained state and
   edit it when needed.
4. Check for an action transition and mark its video time, or explicitly select
   `no_action_switch` when no transition occurs.

`depends_on_segment_ids` is intentionally hidden and is not submitted by the
current UI. Existing values in loaded records are preserved during saves.

Every saved segment must also complete the action-switch annotation using one
of these mutually exclusive forms:

```json
{
  "action_switch_times": [1.275, 3.84],
  "no_action_switch": false
}
```

or:

```json
{
  "action_switch_times": [],
  "no_action_switch": true
}
```

Times are measured in seconds from the beginning of the segment video, rounded
to millisecond precision, de-duplicated, and stored in ascending order. The UI
validates that marked times are inside the video duration when metadata is
available. Existing annotation files that do not contain either decision are
treated as incomplete and must be reviewed again.

Open <http://127.0.0.1:8888> in a browser after the server starts.

By default, annotations are written outside `meta/` to a repository-relative,
versioned path:

```text
annotation_output/<input_stem>_annotated_YYYYMMDDTHHMMSS_microseconds_episodes_START-END.jsonl
```

The timestamp is created when the server starts, so a new annotation session
does not overwrite an earlier export. Repeated saves of the same range in one
session safely update that range's file. Use `--output` to choose the base
destination; the range suffix is still added:

```bash
python app.py input.jsonl videos --output annotations.jsonl
```

At startup, the newest compatible full snapshot is loaded first, followed by
newer range snapshots in chronological order. This restores work saved by
different assignments. For compatibility, the old
`<input_stem>_annotated.jsonl` beside the input is used when no versioned full
output exists. Records marked with `"annotated": true` are restored so
annotation can continue from the previous session. Exports are written
atomically: the previous complete range file remains intact if serialization
or disk writing fails.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Data

Datasets, generated annotations, and video files are not included in this
repository. The contents of `meta/` and generated files in `annotation_output/`
are ignored by Git except for their README files; copy or link the ELMA data
into `meta/` after cloning.
