import json
import tempfile
import unittest
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import app as annotation_app


class DefaultPathTests(unittest.TestCase):
    def test_elma_defaults_are_relative_to_the_application(self):
        app_root = Path(annotation_app.__file__).resolve().parent

        self.assertEqual(annotation_app.APP_ROOT, app_root)
        self.assertEqual(annotation_app.DEFAULT_META_ROOT, app_root / "meta")
        self.assertEqual(
            annotation_app.DEFAULT_ELMA_JSONL,
            app_root / "meta" / "test_anaphora_per_category4_full_episodes.jsonl",
        )
        self.assertEqual(
            annotation_app.DEFAULT_ELMA_VIDEO_ROOT,
            app_root / "meta" / "test_videos",
        )
        self.assertEqual(
            annotation_app.DEFAULT_OUTPUT_ROOT,
            app_root / "annotation_output",
        )

    def test_timestamped_output_is_versioned_and_latest_can_be_found(self):
        first_time = datetime(2026, 7, 14, 12, 30, 45, 123456)
        second_time = datetime(2026, 7, 14, 12, 31, 5, 654321)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            first = annotation_app.timestamped_output_path(
                "/data/input.jsonl", output_root, first_time
            )
            first.write_text("{}\n", encoding="utf-8")
            second = annotation_app.timestamped_output_path(
                "/data/input.jsonl", output_root, second_time
            )
            second.write_text("{}\n", encoding="utf-8")
            (output_root / "other_annotated_20990101T000000_000000.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )

            self.assertEqual(
                first.name,
                "input_annotated_20260714T123045_123456.jsonl",
            )
            self.assertEqual(
                annotation_app.latest_timestamped_output(
                    "/data/input.jsonl", output_root
                ),
                second,
            )

    def test_timestamped_output_uses_a_sequence_on_collision(self):
        fixed_time = datetime(2026, 7, 14, 12, 30, 45, 123456)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            first = annotation_app.timestamped_output_path(
                "/data/input.jsonl", output_root, fixed_time
            )
            first.touch()
            second = annotation_app.timestamped_output_path(
                "/data/input.jsonl", output_root, fixed_time
            )

            self.assertEqual(
                second.name,
                "input_annotated_20260714T123045_123456_002.jsonl",
            )


class ActionSwitchValidationTests(unittest.TestCase):
    def test_normalize_sorts_rounds_and_deduplicates(self):
        result = annotation_app.normalize_action_switch_times(
            [2, "1.2344", 1.23449, 3.0004]
        )
        self.assertEqual(result, [1.234, 2.0, 3.0])

    def test_normalize_rejects_invalid_values(self):
        invalid_values = ([0], [-1], [True], ["bad"], [float("inf")], "1.0")
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    annotation_app.normalize_action_switch_times(values)

    def test_annotation_requires_times_or_explicit_no_switch(self):
        self.assertFalse(annotation_app.is_action_switch_annotated({}))
        self.assertTrue(
            annotation_app.is_action_switch_annotated(
                {"action_switch_times": [1.5], "no_action_switch": False}
            )
        )
        self.assertTrue(
            annotation_app.is_action_switch_annotated(
                {"action_switch_times": [], "no_action_switch": True}
            )
        )
        self.assertFalse(
            annotation_app.is_action_switch_annotated(
                {"action_switch_times": [1.5], "no_action_switch": True}
            )
        )


class AnnotationApiTests(unittest.TestCase):
    def setUp(self):
        annotation_app.app.config.update(TESTING=True)
        annotation_app.DATA["entries"] = [
            {
                "episode_id": "episode-1",
                "target_segment_id": 1,
                "target_video_name": "one.mp4",
                "target_caption": "first action",
            },
            {
                "episode_id": "episode-1",
                "target_segment_id": 2,
                "target_video_name": "two.mp4",
                "target_caption": "second action",
            },
        ]
        annotation_app.DATA["episodes"] = OrderedDict({"episode-1": [0, 1]})
        annotation_app.DATA["episode_ids"] = ["episode-1"]
        annotation_app.DATA["modified"] = {}
        annotation_app.DATA["available_videos"] = set()
        annotation_app.DATA["video_paths"] = {}
        annotation_app.DATA["output_path"] = ""
        self.client = annotation_app.app.test_client()

    def test_page_contains_action_switch_controls(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Action switch times", response.data)
        self.assertIn(b"No action switch in this video", response.data)

    def test_episode_update_is_atomic_when_switch_annotation_is_missing(self):
        response = self.client.post(
            "/api/update_episode",
            json={
                "segments": [
                    {
                        "index": 0,
                        "action_switch_times": [2.5],
                        "no_action_switch": False,
                    },
                    {
                        "index": 1,
                        "action_switch_times": [],
                        "no_action_switch": False,
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])
        self.assertEqual(annotation_app.DATA["modified"], {})

    def test_episode_update_accepts_times_and_explicit_no_switch(self):
        response = self.client.post(
            "/api/update_episode",
            json={
                "segments": [
                    {
                        "index": 0,
                        "action_switch_times": [3.2, 1.1, 3.2],
                        "no_action_switch": False,
                    },
                    {
                        "index": 1,
                        "action_switch_times": [],
                        "no_action_switch": True,
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["updated"], 2)
        self.assertEqual(
            annotation_app.DATA["modified"][0]["action_switch_times"],
            [1.1, 3.2],
        )
        self.assertTrue(annotation_app.DATA["modified"][1]["no_action_switch"])

        info = self.client.get("/api/info").get_json()
        self.assertEqual(info["action_switch_annotated_count"], 2)

    def test_completed_episode_exports_valid_jsonl_in_original_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = annotation_app.timestamped_output_path(
                "/data/input.jsonl",
                temporary_directory,
                datetime(2026, 7, 14, 13, 0, 0, 1),
            )
            annotation_app.DATA["output_path"] = str(output_path)

            update_response = self.client.post(
                "/api/update_episode",
                json={
                    "segments": [
                        {
                            "index": 0,
                            "action_switch_times": [3.2, 1.1, 3.2],
                            "no_action_switch": False,
                        },
                        {
                            "index": 1,
                            "action_switch_times": [],
                            "no_action_switch": True,
                        },
                    ]
                },
            )
            save_response = self.client.post("/api/save")

            self.assertEqual(update_response.status_code, 200)
            self.assertEqual(save_response.status_code, 200)
            save_result = save_response.get_json()
            self.assertTrue(save_result["ok"])
            self.assertTrue(save_result["complete"])
            self.assertEqual(save_result["total_entries"], 2)
            self.assertEqual(save_result["modified_count"], 2)
            self.assertEqual(save_result["action_switch_annotated_count"], 2)
            self.assertEqual(save_result["saved_to"], str(output_path))

            exported_entries = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(exported_entries), 2)
            self.assertEqual(
                [entry["target_segment_id"] for entry in exported_entries],
                [1, 2],
            )
            self.assertEqual(exported_entries[0]["action_switch_times"], [1.1, 3.2])
            self.assertFalse(exported_entries[0]["no_action_switch"])
            self.assertEqual(exported_entries[1]["action_switch_times"], [])
            self.assertTrue(exported_entries[1]["no_action_switch"])
            self.assertTrue(all(entry["annotated"] for entry in exported_entries))
            self.assertEqual(list(Path(temporary_directory).glob("*.tmp")), [])

    def test_failed_atomic_export_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "annotations.jsonl"
            original_contents = '{"existing": true}\n'
            output_path.write_text(original_contents, encoding="utf-8")

            with self.assertRaises(TypeError):
                annotation_app.write_jsonl_atomic(
                    output_path,
                    [{"not_json_serializable": object()}],
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                original_contents,
            )
            self.assertEqual(list(Path(temporary_directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
