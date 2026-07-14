# Anaphoric Annotation Tool

A lightweight Flask web interface for reviewing motion episodes alongside
their videos and correcting event-anaphora annotations.

## Features

- Browse and search episodes and captions.
- Play the video associated with each motion segment.
- Edit `event_anaphora`, `depends_on_segment_ids`, and `keep_body_parts`.
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

### Flat JSONL and video directory

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

## Data

Datasets and video files are not included in this repository.
