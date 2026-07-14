#!/usr/bin/env python3
"""Anaphora annotation tool - Flask backend."""
'''
Usage:
python3 app.py --host 0.0.0.0 --port 8888
python3 app.py meta/input.jsonl meta/test_videos --host 0.0.0.0 --port 8888
'''
import argparse
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file, abort

app = Flask(__name__)

DATA = {
    "entries": [],           # all entries in original order
    "episodes": OrderedDict(),  # episode_id -> list of entry indices
    "episode_ids": [],       # ordered list of episode_ids
    "modified": {},          # index -> modified entry
    "jsonl_path": "",
    "kwargs_root": "",
    "output_path": "",
    "video_root": "",
    "available_videos": set(),  # video filenames in video_root
    "video_paths": {},           # video token/name -> absolute file path
}

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_META_ROOT = APP_ROOT / "meta"
DEFAULT_ELMA_JSONL = DEFAULT_META_ROOT / "test_anaphora_per_category4_full_episodes.jsonl"
DEFAULT_ELMA_VIDEO_ROOT = DEFAULT_META_ROOT / "test_videos"
DEFAULT_OUTPUT_ROOT = APP_ROOT / "annotation_output"
DEFAULT_FLOWMDM_ROOT = APP_ROOT / "flowmdm_ours_results"
DEFAULT_KWARGS_ROOT = DEFAULT_FLOWMDM_ROOT / "flowmdm_results"
EPISODES_PER_PAGE = 100


def load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamped_output_path(input_path, output_root=DEFAULT_OUTPUT_ROOT, now=None):
    """Return a collision-safe, timestamped output path for a new app session."""
    output_root = Path(output_root)
    timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S_%f")
    prefix = f"{Path(input_path).stem}_annotated_{timestamp}"
    candidate = output_root / f"{prefix}.jsonl"

    def path_version_exists(path):
        if path.exists() or not output_root.is_dir():
            return path.exists()
        range_prefix = f"{path.stem}_episodes_"
        return any(
            existing.is_file()
            and existing.name.startswith(range_prefix)
            and existing.suffix == ".jsonl"
            for existing in output_root.iterdir()
        )

    sequence = 1
    while path_version_exists(candidate):
        sequence += 1
        candidate = output_root / f"{prefix}_{sequence:03d}.jsonl"
    return candidate


def timestamped_output_version(path, input_path):
    """Parse the sortable timestamp and collision sequence of a full output."""
    stem = re.escape(Path(input_path).stem)
    pattern = re.compile(
        rf"^{stem}_annotated_(\d{{8}}T\d{{6}}_\d{{6}})(?:_(\d{{3}}))?\.jsonl$"
    )
    match = pattern.match(Path(path).name)
    if not match:
        return None
    try:
        timestamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S_%f")
    except ValueError:
        return None
    return timestamp, int(match.group(2) or 1)


def latest_timestamped_output(input_path, output_root=DEFAULT_OUTPUT_ROOT):
    """Find the newest valid full timestamped output for an input file."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return None

    candidates = []
    for path in output_root.iterdir():
        if not path.is_file():
            continue
        version = timestamped_output_version(path, input_path)
        if version is None:
            continue
        candidates.append((*version, path))

    return max(candidates, default=(None, None, None))[-1]


def timestamped_range_outputs(input_path, output_root=DEFAULT_OUTPUT_ROOT):
    """Return timestamped range outputs in chronological order."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []

    stem = re.escape(Path(input_path).stem)
    pattern = re.compile(
        rf"^{stem}_annotated_(\d{{8}}T\d{{6}}_\d{{6}})"
        rf"(?:_(\d{{3}}))?_episodes_(\d+)-(\d+)\.jsonl$"
    )
    candidates = []
    for path in output_root.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            timestamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S_%f")
        except ValueError:
            continue
        candidates.append({
            "path": path,
            "range_start": int(match.group(3)),
            "range_end": int(match.group(4)),
            "version": (timestamp, int(match.group(2) or 1)),
        })

    return sorted(
        candidates,
        key=lambda candidate: (*candidate["version"], candidate["path"].name),
    )


def legacy_output_path(input_path):
    """Return the pre-versioning output location used by older releases."""
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_annotated.jsonl")


def range_output_path(output_path, range_start, range_end, total_episodes):
    """Add a zero-padded inclusive episode range to an output filename."""
    output_path = Path(output_path)
    width = max(3, len(str(total_episodes)))
    suffix = output_path.suffix or ".jsonl"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(
        f"{stem}_episodes_{range_start:0{width}d}-{range_end:0{width}d}{suffix}"
    )


def write_jsonl_atomic(path, entries):
    """Write JSONL without exposing a truncated or partially written result."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for entry in entries:
                temporary_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def sample_key(index):
    """FlowMDM outputs use 00-99 with two digits, then plain decimal."""
    return f"{index:02d}" if index < 100 else str(index)


def first_positive_index(values):
    for i, value in enumerate(values or []):
        try:
            if int(value) > 0:
                return i
        except (TypeError, ValueError):
            continue
    return None


def build_flowmdm_entries(humanml_path, kwargs_root, video_root):
    """Build annotation entries by aligning dataset rows, kwargs files, and renders."""
    samples = load_json(humanml_path)
    entries = []

    for sample_index, sample in enumerate(samples):
        key = sample_key(sample_index)
        kwargs_path = os.path.join(kwargs_root, f"{key}_kwargs.json")
        render_video = os.path.join(video_root, key, "tpv.mp4")
        video_name = f"{key}_tpv.mp4"

        kwargs_y = {}
        if os.path.exists(kwargs_path):
            kwargs_data = load_json(kwargs_path)
            kwargs_y = kwargs_data.get("y", kwargs_data)

        texts = sample.get("text") or kwargs_y.get("text") or []
        kwargs_texts = kwargs_y.get("text") or []
        lengths = sample.get("lengths") or kwargs_y.get("lengths") or []
        target_idx = first_positive_index(lengths)

        if os.path.exists(render_video):
            DATA["available_videos"].add(video_name)
            DATA["video_paths"][video_name] = render_video

        episode_id = sample.get("id") or kwargs_y.get("id") or key
        for segment_index, text in enumerate(texts):
            segment_id = segment_index + 1
            kwargs_caption = kwargs_texts[segment_index] if segment_index < len(kwargs_texts) else ""
            entries.append({
                "episode_id": episode_id,
                "sample_index": sample_index,
                "sample_key": key,
                "source_id": sample.get("source_id", sample.get("id", "")),
                "target_segment_id": segment_id,
                "target_video_name": video_name,
                "target_caption": text,
                "flowmdm_caption": kwargs_caption,
                "segment_length": lengths[segment_index] if segment_index < len(lengths) else "",
                "is_target_segment": target_idx == segment_index,
                "history_text": sample.get("history_text", []),
                "scenario": sample.get("scenario", ""),
                "kwargs_path": kwargs_path if os.path.exists(kwargs_path) else "",
                "render_video_path": render_video if os.path.exists(render_video) else "",
                "event_anaphora": target_idx == segment_index,
                "depends_on_segment_ids": [],
                "keep_body_parts": [],
                "action_switch_times": [],
                "no_action_switch": False,
            })

    return entries


def build_episode_index(entries):
    episodes = OrderedDict()
    for i, e in enumerate(entries):
        ep = e.get("episode_id", "unknown")
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(i)
    return episodes


def parse_episode_range(args, total_episodes):
    """Parse a 1-based inclusive episode assignment range."""
    if total_episodes == 0:
        return 1, 0

    try:
        range_start = int(args.get("range_start", 1))
        range_end = int(args.get("range_end", total_episodes))
    except (TypeError, ValueError) as exc:
        raise ValueError("episode range values must be integers") from exc

    if not 1 <= range_start <= range_end <= total_episodes:
        raise ValueError(
            "episode range must satisfy "
            f"1 <= range_start <= range_end <= {total_episodes}"
        )
    return range_start, range_end


def episode_range_entry_indices(episodes, episode_ids, range_start, range_end):
    """Return original entry indices for an inclusive episode range."""
    return sorted(
        index
        for episode_id in episode_ids[range_start - 1:range_end]
        for index in episodes[episode_id]
    )


def restore_output_entries(original_entries, modified_entries, output_entries, indices):
    """Restore one full or partial output after validating record alignment."""
    if len(output_entries) != len(indices):
        return False

    def identity(entry):
        return (
            entry.get("episode_id"),
            entry.get("target_segment_id"),
            entry.get("target_video_name"),
        )

    if any(
        identity(original_entries[index]) != identity(output_entry)
        for index, output_entry in zip(indices, output_entries)
    ):
        return False

    for index, output_entry in zip(indices, output_entries):
        if output_entry.get("annotated"):
            modified_entries[index] = output_entry
        else:
            modified_entries.pop(index, None)
    return True


def resolve_video_path(video_name):
    """Find videos either from explicit FlowMDM mapping or a flat video root."""
    if video_name in DATA["video_paths"]:
        return DATA["video_paths"][video_name]
    if video_name in DATA["available_videos"]:
        return os.path.join(DATA["video_root"], video_name)
    return None


def normalize_action_switch_times(values):
    """Validate, round, de-duplicate, and sort action-switch times in seconds."""
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("action_switch_times must be a JSON array")

    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("action-switch times must be numbers")
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("action-switch times must be numbers") from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("action-switch times must be greater than 0 seconds")
        seconds = round(seconds, 3)
        if seconds <= 0:
            raise ValueError("action-switch times must be greater than 0 seconds")
        normalized.append(seconds)

    return sorted(set(normalized))


def action_switch_annotation(entry):
    """Return normalized switch times and the explicit no-switch decision."""
    try:
        times = normalize_action_switch_times(entry.get("action_switch_times", []))
    except ValueError:
        times = []
    no_switch = entry.get("no_action_switch") is True
    return times, no_switch


def is_action_switch_annotated(entry):
    """A completed annotation has timestamps or an explicit no-switch decision."""
    try:
        times = normalize_action_switch_times(entry.get("action_switch_times", []))
    except ValueError:
        return False
    no_switch = entry.get("no_action_switch") is True
    return (no_switch and not times) or (not no_switch and bool(times))


def apply_action_switch_annotation(entry, payload):
    """Apply and validate required action-switch fields on an entry copy."""
    raw_times = payload.get(
        "action_switch_times", entry.get("action_switch_times", [])
    )
    raw_no_switch = payload.get(
        "no_action_switch", entry.get("no_action_switch", False)
    )
    if not isinstance(raw_no_switch, bool):
        raise ValueError("no_action_switch must be a boolean")

    times = normalize_action_switch_times(raw_times)
    if raw_no_switch and times:
        raise ValueError(
            "choose either action-switch times or no_action_switch, not both"
        )
    if not raw_no_switch and not times:
        raise ValueError(
            "mark at least one action-switch time or select no_action_switch"
        )

    entry["action_switch_times"] = times
    entry["no_action_switch"] = raw_no_switch
    return entry


def apply_target_caption(entry, payload):
    """Apply a non-empty editable target caption when it is submitted."""
    if "target_caption" not in payload:
        return entry

    caption = payload["target_caption"]
    if not isinstance(caption, str):
        raise ValueError("target_caption must be a string")
    caption = caption.strip()
    if not caption:
        raise ValueError("target_caption must not be empty")
    entry["target_caption"] = caption
    return entry


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    current_entries = [
        DATA["modified"].get(i, entry) for i, entry in enumerate(DATA["entries"])
    ]
    return jsonify({
        "total_entries": len(DATA["entries"]),
        "total_episodes": len(DATA["episode_ids"]),
        "modified_count": len(DATA["modified"]),
        "action_switch_annotated_count": sum(
            is_action_switch_annotated(entry) for entry in current_entries
        ),
        "output_path": DATA["output_path"],
    })


@app.route("/api/episode_list")
def api_episode_list():
    """Return a filtered page of episodes with summary and pagination info."""
    q = request.args.get("q", "").strip().lower()
    filter_type = request.args.get("filter", "all")
    try:
        requested_page = int(request.args.get("page", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "page must be a positive integer"}), 400
    if requested_page < 1:
        return jsonify({"error": "page must be a positive integer"}), 400

    try:
        range_start, range_end = parse_episode_range(
            request.args, len(DATA["episode_ids"])
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    results = []
    range_episode_ids = DATA["episode_ids"][range_start - 1:range_end]
    for episode_number, ep_id in enumerate(range_episode_ids, start=range_start):
        indices = DATA["episodes"][ep_id]
        has_modified = any(i in DATA["modified"] for i in indices)
        all_modified = all(i in DATA["modified"] for i in indices)
        switch_annotated_count = sum(
            is_action_switch_annotated(DATA["modified"].get(i, DATA["entries"][i]))
            for i in indices
        )
        all_action_switch_annotated = switch_annotated_count == len(indices)
        has_video = any(
            resolve_video_path(DATA["entries"][i].get("target_video_name", "")) is not None
            for i in indices
        )

        if filter_type == "modified" and not has_modified:
            continue
        if filter_type == "unmodified" and all_modified:
            continue
        if filter_type == "has_video" and not has_video:
            continue
        if filter_type == "switch_pending" and all_action_switch_annotated:
            continue

        if q:
            match = False
            if q in ep_id.lower():
                match = True
            else:
                for i in indices:
                    e = DATA["modified"].get(i, DATA["entries"][i])
                    cap = (e.get("target_caption") or "").lower()
                    cat = (e.get("babel_category") or "").lower()
                    if q in cap or q in cat:
                        match = True
                        break
            if not match:
                continue

        results.append({
            "episode_id": ep_id,
            "episode_number": episode_number,
            "num_segments": len(indices),
            "first_index": indices[0],
            "has_modified": has_modified,
            "all_modified": all_modified,
            "switch_annotated_count": switch_annotated_count,
            "all_action_switch_annotated": all_action_switch_annotated,
            "has_video": has_video,
            "category": DATA["entries"][indices[0]].get("babel_category", ""),
            "sample_index": DATA["entries"][indices[0]].get("sample_index"),
            "sample_key": DATA["entries"][indices[0]].get("sample_key", ""),
        })

    total_items = len(results)
    total_pages = max(1, math.ceil(total_items / EPISODES_PER_PAGE))
    page = min(requested_page, total_pages)
    start = (page - 1) * EPISODES_PER_PAGE
    page_results = results[start:start + EPISODES_PER_PAGE]
    return jsonify({
        "results": page_results,
        "pagination": {
            "page": page,
            "page_size": EPISODES_PER_PAGE,
            "total_items": total_items,
            "total_pages": total_pages,
        },
        "range": {
            "start": range_start,
            "end": range_end,
            "total_items": len(range_episode_ids),
        },
    })


@app.route("/api/episode/<episode_id>")
def api_episode(episode_id):
    """Return all segments of an episode with current (possibly modified) values."""
    if episode_id not in DATA["episodes"]:
        abort(404)

    try:
        range_start, range_end = parse_episode_range(
            request.args, len(DATA["episode_ids"])
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    range_episode_ids = DATA["episode_ids"][range_start - 1:range_end]
    if episode_id not in range_episode_ids:
        abort(404, "Episode is outside the selected assignment range")

    indices = DATA["episodes"][episode_id]
    segments = []
    for i in indices:
        entry = DATA["entries"][i]
        mod = DATA["modified"].get(i)
        current = mod if mod else entry
        video_name = entry.get("target_video_name", "")
        switch_times, no_switch = action_switch_annotation(current)
        segments.append({
            "index": i,
            "target_segment_id": entry.get("target_segment_id"),
            "target_video_name": video_name,
            "target_caption": current.get("target_caption", ""),
            "has_video": resolve_video_path(video_name) is not None,
            "is_modified": i in DATA["modified"],
            # Editable fields - current values
            "event_anaphora": current.get("event_anaphora", False),
            "depends_on_segment_ids": current.get("depends_on_segment_ids", []),
            "keep_body_parts": current.get("keep_body_parts", []),
            "action_switch_times": switch_times,
            "no_action_switch": no_switch,
            "action_switch_annotated": is_action_switch_annotated(current),
            # Context fields
            "anaphoric_expression_in_target": current.get("anaphoric_expression_in_target", ""),
            "antecedent_expression": current.get("antecedent_expression", ""),
            "dependency_type": current.get("dependency_type", ""),
            "reason": current.get("reason", ""),
            "babel_category": entry.get("babel_category", ""),
            "babel_category_id": entry.get("babel_category_id", ""),
            "sample_index": entry.get("sample_index"),
            "sample_key": entry.get("sample_key", ""),
            "source_id": entry.get("source_id", ""),
            "scenario": entry.get("scenario", ""),
            "flowmdm_caption": entry.get("flowmdm_caption", ""),
            "segment_length": entry.get("segment_length", ""),
            "is_target_segment": entry.get("is_target_segment", False),
            "kwargs_path": entry.get("kwargs_path", ""),
            "render_video_path": entry.get("render_video_path", ""),
        })
    # Find episode position
    ep_idx = DATA["episode_ids"].index(episode_id)
    range_idx = range_episode_ids.index(episode_id)
    prev_ep = range_episode_ids[range_idx - 1] if range_idx > 0 else None
    next_ep = (
        range_episode_ids[range_idx + 1]
        if range_idx < len(range_episode_ids) - 1
        else None
    )
    return jsonify({
        "episode_id": episode_id,
        "episode_index": ep_idx,
        "total_episodes": len(DATA["episode_ids"]),
        "range_position": range_idx + 1,
        "range_total": len(range_episode_ids),
        "range_start": range_start,
        "range_end": range_end,
        "segments": segments,
        "prev_episode": prev_ep,
        "next_episode": next_ep,
    })


@app.route("/api/video_by_name/<video_name>")
def api_video_by_name(video_name):
    vpath = resolve_video_path(video_name)
    if not vpath:
        abort(404, f"Video not found: {video_name}")
    return send_file(vpath, mimetype="video/mp4")


@app.route("/api/update", methods=["POST"])
def api_update():
    """Update editable fields for a single entry."""
    data = request.get_json(silent=True) or {}
    idx = data.get("index")
    if (
        isinstance(idx, bool)
        or not isinstance(idx, int)
        or idx < 0
        or idx >= len(DATA["entries"])
    ):
        abort(400)
    # Start from existing modified version if present, otherwise from original
    base = DATA["modified"].get(idx, DATA["entries"][idx])
    modified_entry = dict(base)
    if "event_anaphora" in data:
        modified_entry["event_anaphora"] = bool(data["event_anaphora"])
    # depends_on_segment_ids editing is intentionally disabled; preserve base value.
    if "keep_body_parts" in data:
        modified_entry["keep_body_parts"] = data["keep_body_parts"]
    try:
        apply_target_caption(modified_entry, data)
        apply_action_switch_annotation(modified_entry, data)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    modified_entry["annotated"] = True
    DATA["modified"][idx] = modified_entry
    return jsonify({"ok": True, "modified_count": len(DATA["modified"])})


@app.route("/api/update_episode", methods=["POST"])
def api_update_episode():
    """Batch update all segments of an episode."""
    data = request.get_json(silent=True) or {}
    segments = data.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return jsonify({"ok": False, "error": "No segments were submitted."}), 400
    pending_updates = []
    errors = []
    for seg in segments:
        if not isinstance(seg, dict):
            errors.append({"index": None, "error": "segment must be an object"})
            continue
        idx = seg.get("index")
        if (
            isinstance(idx, bool)
            or not isinstance(idx, int)
            or idx < 0
            or idx >= len(DATA["entries"])
        ):
            errors.append({"index": idx, "error": "invalid segment index"})
            continue
        base = DATA["modified"].get(idx, DATA["entries"][idx])
        modified_entry = dict(base)
        if "event_anaphora" in seg:
            modified_entry["event_anaphora"] = bool(seg["event_anaphora"])
        # depends_on_segment_ids editing is intentionally disabled; preserve base value.
        if "keep_body_parts" in seg:
            modified_entry["keep_body_parts"] = seg["keep_body_parts"]
        try:
            apply_target_caption(modified_entry, seg)
            apply_action_switch_annotation(modified_entry, seg)
        except ValueError as exc:
            errors.append({"index": idx, "error": str(exc)})
            continue
        modified_entry["annotated"] = True
        pending_updates.append((idx, modified_entry))

    if errors:
        return jsonify({
            "ok": False,
            "error": "One or more segment annotations are invalid.",
            "details": errors,
        }), 400

    for idx, modified_entry in pending_updates:
        DATA["modified"][idx] = modified_entry
    return jsonify({
        "ok": True,
        "updated": len(pending_updates),
        "modified_count": len(DATA["modified"]),
    })


@app.route("/api/delete_annotation", methods=["POST"])
def api_delete_annotation():
    data = request.json
    idx = data.get("index")
    if idx is not None and idx in DATA["modified"]:
        del DATA["modified"][idx]
    return jsonify({"ok": True, "modified_count": len(DATA["modified"])})


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(silent=True) or {}
    total_episodes = len(DATA["episode_ids"])
    try:
        range_start, range_end = parse_episode_range(data, total_episodes)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    range_episode_ids = DATA["episode_ids"][range_start - 1:range_end]
    range_indices = episode_range_entry_indices(
        DATA["episodes"], DATA["episode_ids"], range_start, range_end
    )
    output_entries = [
        DATA["modified"].get(index, DATA["entries"][index])
        for index in range_indices
    ]
    modified_count = sum(index in DATA["modified"] for index in range_indices)
    action_switch_annotated_count = sum(
        is_action_switch_annotated(entry) for entry in output_entries
    )
    output_path = str(
        range_output_path(
            DATA["output_path"], range_start, range_end, total_episodes
        )
    )
    try:
        write_jsonl_atomic(output_path, output_entries)
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"Export failed: {exc}"}), 500

    return jsonify({
        "ok": True,
        "saved_to": output_path,
        "modified_count": modified_count,
        "total_episodes": len(range_episode_ids),
        "total_entries": len(output_entries),
        "action_switch_annotated_count": action_switch_annotated_count,
        "complete": action_switch_annotated_count == len(output_entries),
        "episode_range": {
            "start": range_start,
            "end": range_end,
        },
    })


def main():
    parser = argparse.ArgumentParser(description="Anaphora annotation tool")
    parser.add_argument(
        "jsonl",
        nargs="?",
        default=str(DEFAULT_ELMA_JSONL),
        help=(
            "Path to input JSONL file, or FlowMDM humanml_test_set_anaphora.json "
            "(default: meta/test_anaphora_per_category4_full_episodes.jsonl "
            "relative to app.py)"
        ),
    )
    parser.add_argument(
        "video_root",
        nargs="?",
        default=str(DEFAULT_ELMA_VIDEO_ROOT),
        help=(
            "Path to video directory, or FlowMDM blender_render_flowmdm directory "
            "(default: meta/test_videos relative to app.py)"
        ),
    )
    parser.add_argument(
        "--kwargs-root",
        default=str(DEFAULT_KWARGS_ROOT),
        help="Path to FlowMDM *_kwargs.json directory",
    )
    parser.add_argument(
        "--flowmdm",
        action="store_true",
        help="Read FlowMDM result layout: humanml JSON + kwargs root + render root",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output JSONL path (default: a timestamped file under "
            "annotation_output relative to app.py)"
        ),
    )
    parser.add_argument("-p", "--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    DATA["jsonl_path"] = os.path.abspath(args.jsonl)
    DATA["video_root"] = os.path.abspath(args.video_root)
    DATA["kwargs_root"] = os.path.abspath(args.kwargs_root)
    flowmdm_mode = args.flowmdm or DATA["jsonl_path"].endswith(".json")

    # Scan available videos for the legacy flat JSONL mode.
    if not flowmdm_mode and os.path.isdir(DATA["video_root"]):
        DATA["available_videos"] = set(os.listdir(DATA["video_root"]))
    print(f"Available videos: {len(DATA['available_videos'])}")

    resume_path = None
    resume_version = None
    range_resume_candidates = []
    if args.output:
        DATA["output_path"] = os.path.abspath(args.output)
        if os.path.exists(DATA["output_path"]):
            resume_path = Path(DATA["output_path"])
    else:
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        resume_path = latest_timestamped_output(DATA["jsonl_path"])
        if resume_path is not None:
            resume_version = timestamped_output_version(
                resume_path, DATA["jsonl_path"]
            )
        if resume_path is None:
            legacy_path = legacy_output_path(DATA["jsonl_path"])
            if legacy_path.exists():
                resume_path = legacy_path
        range_resume_candidates = timestamped_range_outputs(DATA["jsonl_path"])
        if resume_version is not None:
            range_resume_candidates = [
                candidate
                for candidate in range_resume_candidates
                if candidate["version"] >= resume_version
            ]
        DATA["output_path"] = str(
            timestamped_output_path(DATA["jsonl_path"])
        )

    # Load entries
    if flowmdm_mode:
        DATA["entries"] = build_flowmdm_entries(
            DATA["jsonl_path"],
            DATA["kwargs_root"],
            DATA["video_root"],
        )
    else:
        DATA["entries"] = load_jsonl(DATA["jsonl_path"])

    # Build episode index before restoring full and partial range outputs.
    DATA["episodes"] = build_episode_index(DATA["entries"])
    DATA["episode_ids"] = list(DATA["episodes"].keys())
    DATA["modified"] = {}

    # Resume from the latest full snapshot, then newer range snapshots.
    if resume_path is not None:
        print(f"Resuming from: {resume_path}")
        output_entries = load_jsonl(resume_path)
        if not restore_output_entries(
            DATA["entries"],
            DATA["modified"],
            output_entries,
            list(range(len(DATA["entries"]))),
        ):
            print(f"Skipped incompatible output: {resume_path}")

    for candidate in range_resume_candidates:
        range_start = candidate["range_start"]
        range_end = candidate["range_end"]
        try:
            parse_episode_range(
                {"range_start": range_start, "range_end": range_end},
                len(DATA["episode_ids"]),
            )
        except ValueError:
            print(f"Skipped invalid range output: {candidate['path']}")
            continue
        indices = episode_range_entry_indices(
            DATA["episodes"], DATA["episode_ids"], range_start, range_end
        )
        output_entries = load_jsonl(candidate["path"])
        if restore_output_entries(
            DATA["entries"], DATA["modified"], output_entries, indices
        ):
            print(f"Resuming range {range_start}-{range_end}: {candidate['path']}")
        else:
            print(f"Skipped incompatible range output: {candidate['path']}")

    if resume_path is not None or range_resume_candidates:
        print(f"Resumed {len(DATA['modified'])} modified entries")

    print(f"Loaded {len(DATA['entries'])} entries, {len(DATA['episode_ids'])} episodes")
    print(f"Output: {DATA['output_path']}")
    print(f"http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
