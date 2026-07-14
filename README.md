# Anaphoric Annotation Tool

A lightweight Flask web interface for reviewing motion episodes alongside
their videos and correcting event-anaphora annotations.

## Features

- Browse and search episodes and captions.
- Play the video associated with each motion segment.
- Edit `event_anaphora`, `depends_on_segment_ids`, and `keep_body_parts`.
- Mark one or more action-switch times directly from the video player.
- Require an explicit `no_action_switch` decision when a video has no switch.
- Save annotations as JSONL and resume from an existing output file.
- Read either a flat JSONL/video layout or FlowMDM result directories.

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

The editable annotation fields may already be present or will use empty
defaults.

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

### FlowMDM results

```bash
python app.py /path/to/humanml_test_set_anaphora.json \
  /path/to/blender_render_flowmdm \
  --kwargs-root /path/to/flowmdm_results \
  --flowmdm \
  --host 127.0.0.1 \
  --port 8888
```

Open <http://127.0.0.1:8888> in a browser after the server starts.

By default, annotations are written next to the input as
`<input_stem>_annotated.jsonl`. Use `--output` to choose another destination:

```bash
python app.py input.jsonl videos --output annotations.jsonl
```

If the output file already exists and has the same number of entries, records
marked with `"annotated": true` are loaded automatically so annotation can
continue from the previous session.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Data

Datasets, generated annotations, and video files are not included in this
repository. The contents of `meta/` are ignored by Git except for its README;
copy or link the ELMA data into that directory after cloning.
