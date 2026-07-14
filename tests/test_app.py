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

    def test_timestamped_output_avoids_existing_range_version(self):
        fixed_time = datetime(2026, 7, 14, 12, 30, 45, 123456)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            existing_range = (
                output_root
                / "input_annotated_20260714T123045_123456_episodes_001-100.jsonl"
            )
            existing_range.touch()

            output = annotation_app.timestamped_output_path(
                "/data/input.jsonl", output_root, fixed_time
            )

            self.assertEqual(
                output.name,
                "input_annotated_20260714T123045_123456_002.jsonl",
            )

    def test_timestamped_range_outputs_are_discovered_in_version_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            names = (
                "input_annotated_20260714T123100_000000_episodes_101-200.jsonl",
                "input_annotated_20260714T123000_000000_episodes_001-100.jsonl",
                "other_annotated_20260714T123200_000000_episodes_201-300.jsonl",
            )
            for name in names:
                (output_root / name).write_text("{}\n", encoding="utf-8")

            discovered = annotation_app.timestamped_range_outputs(
                "/data/input.jsonl", output_root
            )

            self.assertEqual(
                [candidate["path"].name for candidate in discovered],
                [names[1], names[0]],
            )
            self.assertEqual(
                [
                    (candidate["range_start"], candidate["range_end"])
                    for candidate in discovered
                ],
                [(1, 100), (101, 200)],
            )


class OutputRestoreTests(unittest.TestCase):
    def setUp(self):
        self.original = [
            {
                "episode_id": f"episode-{index}",
                "target_segment_id": 1,
                "target_video_name": f"video-{index}.mp4",
            }
            for index in range(1, 4)
        ]

    def test_partial_output_restores_and_can_clear_earlier_modifications(self):
        modified = {0: {**self.original[0], "annotated": True}}
        partial_output = [
            {**self.original[1], "annotated": True, "target_caption": "edited"},
            dict(self.original[2]),
        ]

        restored = annotation_app.restore_output_entries(
            self.original,
            modified,
            partial_output,
            [1, 2],
        )

        self.assertTrue(restored)
        self.assertIn(0, modified)
        self.assertEqual(modified[1]["target_caption"], "edited")
        self.assertNotIn(2, modified)

        cleared = annotation_app.restore_output_entries(
            self.original,
            modified,
            [dict(self.original[1])],
            [1],
        )
        self.assertTrue(cleared)
        self.assertNotIn(1, modified)

    def test_incompatible_partial_output_is_rejected_without_changes(self):
        modified = {0: {**self.original[0], "annotated": True}}
        mismatched = [{**self.original[1], "episode_id": "wrong"}]

        restored = annotation_app.restore_output_entries(
            self.original,
            modified,
            mismatched,
            [1],
        )

        self.assertFalse(restored)
        self.assertEqual(list(modified), [0])


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


class CaptionValidationTests(unittest.TestCase):
    def test_caption_is_trimmed_and_saved(self):
        entry = {"target_caption": "original"}

        annotation_app.apply_target_caption(
            entry,
            {"target_caption": "  revised caption  "},
        )

        self.assertEqual(entry["target_caption"], "revised caption")

    def test_caption_rejects_empty_and_non_string_values(self):
        for caption in ("", "   ", None, 123, []):
            with self.subTest(caption=caption):
                with self.assertRaises(ValueError):
                    annotation_app.apply_target_caption(
                        {"target_caption": "original"},
                        {"target_caption": caption},
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

    def configure_episode_count(self, count):
        entries = []
        episodes = OrderedDict()
        for index in range(count):
            episode_id = f"episode-{index + 1:03d}"
            entries.append({
                "episode_id": episode_id,
                "target_segment_id": 1,
                "target_video_name": f"{episode_id}.mp4",
                "target_caption": f"caption {index + 1:03d}",
            })
            episodes[episode_id] = [index]
        annotation_app.DATA["entries"] = entries
        annotation_app.DATA["episodes"] = episodes
        annotation_app.DATA["episode_ids"] = list(episodes)
        annotation_app.DATA["modified"] = {}

    def test_page_contains_action_switch_controls(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Action switch times", response.data)
        self.assertIn(b"No action switch in this video", response.data)
        self.assertIn(b'data-field="target_caption"', response.data)
        self.assertIn(b'id="prevPageBtn"', response.data)
        self.assertIn(b'id="nextPageBtn"', response.data)
        self.assertIn(b'id="rangeStartInput"', response.data)
        self.assertIn(b'id="rangeEndInput"', response.data)
        self.assertIn(b"Episode assignment range", response.data)
        self.assertIn(b"Save Assigned Range", response.data)
        workflow_markers = [
            response.data.index(f'data-workflow-step="{step}"'.encode())
            for step in range(1, 5)
        ]
        self.assertEqual(workflow_markers, sorted(workflow_markers))
        self.assertNotIn(b'data-field="depends_on"', response.data)
        self.assertIn(b"Maintained state", response.data)
        self.assertIn(b"Body parts to maintain", response.data)
        self.assertIn(b"Caption check", response.data)

    def test_episode_list_returns_100_items_per_page(self):
        self.configure_episode_count(250)

        first = self.client.get("/api/episode_list?page=1").get_json()
        second = self.client.get("/api/episode_list?page=2").get_json()
        third = self.client.get("/api/episode_list?page=3").get_json()

        self.assertEqual(annotation_app.EPISODES_PER_PAGE, 100)
        self.assertEqual(len(first["results"]), 100)
        self.assertEqual(len(second["results"]), 100)
        self.assertEqual(len(third["results"]), 50)
        self.assertEqual(first["results"][0]["episode_id"], "episode-001")
        self.assertEqual(second["results"][0]["episode_id"], "episode-101")
        self.assertEqual(third["results"][-1]["episode_id"], "episode-250")
        self.assertEqual(
            third["pagination"],
            {
                "page": 3,
                "page_size": 100,
                "total_items": 250,
                "total_pages": 3,
            },
        )

        beyond_last = self.client.get("/api/episode_list?page=99").get_json()
        self.assertEqual(beyond_last["pagination"]["page"], 3)
        self.assertEqual(len(beyond_last["results"]), 50)

    def test_episode_list_paginates_filtered_and_searched_results(self):
        self.configure_episode_count(205)
        for index in range(150):
            annotation_app.DATA["modified"][index] = {
                **annotation_app.DATA["entries"][index],
                "annotated": True,
            }

        searched = self.client.get(
            "/api/episode_list?q=caption&page=2"
        ).get_json()
        modified = self.client.get(
            "/api/episode_list?filter=modified&page=2"
        ).get_json()

        self.assertEqual(searched["pagination"]["total_items"], 205)
        self.assertEqual(searched["pagination"]["total_pages"], 3)
        self.assertEqual(searched["results"][0]["episode_id"], "episode-101")
        self.assertEqual(modified["pagination"]["total_items"], 150)
        self.assertEqual(modified["pagination"]["total_pages"], 2)
        self.assertEqual(len(modified["results"]), 50)

    def test_episode_list_is_limited_to_inclusive_assignment_range(self):
        self.configure_episode_count(500)

        first = self.client.get(
            "/api/episode_list?range_start=200&range_end=300&page=1"
        ).get_json()
        second = self.client.get(
            "/api/episode_list?range_start=200&range_end=300&page=2"
        ).get_json()

        self.assertEqual(first["range"], {"start": 200, "end": 300, "total_items": 101})
        self.assertEqual(first["pagination"]["total_items"], 101)
        self.assertEqual(first["pagination"]["total_pages"], 2)
        self.assertEqual(len(first["results"]), 100)
        self.assertEqual(len(second["results"]), 1)
        self.assertEqual(first["results"][0]["episode_id"], "episode-200")
        self.assertEqual(first["results"][0]["episode_number"], 200)
        self.assertEqual(first["results"][-1]["episode_id"], "episode-299")
        self.assertEqual(second["results"][0]["episode_id"], "episode-300")
        self.assertEqual(second["results"][0]["episode_number"], 300)

    def test_search_only_uses_episodes_inside_assignment_range(self):
        self.configure_episode_count(500)

        outside = self.client.get(
            "/api/episode_list?range_start=200&range_end=300&q=caption%20199"
        ).get_json()
        inside = self.client.get(
            "/api/episode_list?range_start=200&range_end=300&q=caption%20250"
        ).get_json()

        self.assertEqual(outside["pagination"]["total_items"], 0)
        self.assertEqual(outside["results"], [])
        self.assertEqual(
            [result["episode_id"] for result in inside["results"]],
            ["episode-250"],
        )

    def test_episode_list_rejects_invalid_pages(self):
        for page in ("0", "-1", "invalid"):
            with self.subTest(page=page):
                response = self.client.get(f"/api/episode_list?page={page}")
                self.assertEqual(response.status_code, 400)
                self.assertIn("positive integer", response.get_json()["error"])

    def test_episode_list_rejects_invalid_assignment_ranges(self):
        self.configure_episode_count(500)
        invalid_queries = (
            "range_start=0&range_end=100",
            "range_start=1&range_end=501",
            "range_start=300&range_end=200",
            "range_start=invalid&range_end=300",
        )
        for query in invalid_queries:
            with self.subTest(query=query):
                response = self.client.get(f"/api/episode_list?{query}")
                self.assertEqual(response.status_code, 400)
                self.assertIn("episode range", response.get_json()["error"])

    def test_episode_detail_reports_full_episode_count(self):
        self.configure_episode_count(250)

        episode = self.client.get("/api/episode/episode-250").get_json()

        self.assertEqual(episode["episode_index"], 249)
        self.assertEqual(episode["total_episodes"], 250)

    def test_episode_navigation_stays_inside_assignment_range(self):
        self.configure_episode_count(500)

        first = self.client.get(
            "/api/episode/episode-200?range_start=200&range_end=300"
        ).get_json()
        middle = self.client.get(
            "/api/episode/episode-250?range_start=200&range_end=300"
        ).get_json()
        last = self.client.get(
            "/api/episode/episode-300?range_start=200&range_end=300"
        ).get_json()
        outside = self.client.get(
            "/api/episode/episode-199?range_start=200&range_end=300"
        )

        self.assertIsNone(first["prev_episode"])
        self.assertEqual(first["next_episode"], "episode-201")
        self.assertEqual(first["range_position"], 1)
        self.assertEqual(first["range_total"], 101)
        self.assertEqual(middle["range_position"], 51)
        self.assertEqual(middle["prev_episode"], "episode-249")
        self.assertEqual(middle["next_episode"], "episode-251")
        self.assertEqual(last["prev_episode"], "episode-299")
        self.assertIsNone(last["next_episode"])
        self.assertEqual(outside.status_code, 404)

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
                        "target_caption": "  revised first action  ",
                        "action_switch_times": [3.2, 1.1, 3.2],
                        "no_action_switch": False,
                    },
                    {
                        "index": 1,
                        "target_caption": "revised second action",
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
        self.assertEqual(
            annotation_app.DATA["modified"][0]["target_caption"],
            "revised first action",
        )
        self.assertTrue(annotation_app.DATA["modified"][1]["no_action_switch"])

        episode = self.client.get("/api/episode/episode-1").get_json()
        self.assertEqual(
            [segment["target_caption"] for segment in episode["segments"]],
            ["revised first action", "revised second action"],
        )
        search = self.client.get("/api/episode_list?q=revised%20second").get_json()
        self.assertEqual(
            [result["episode_id"] for result in search["results"]],
            ["episode-1"],
        )

        info = self.client.get("/api/info").get_json()
        self.assertEqual(info["action_switch_annotated_count"], 2)

    def test_episode_update_preserves_hidden_dependency_ids(self):
        annotation_app.DATA["entries"][0]["depends_on_segment_ids"] = [7, 8]
        annotation_app.DATA["entries"][1]["depends_on_segment_ids"] = [9]

        response = self.client.post(
            "/api/update_episode",
            json={
                "segments": [
                    {
                        "index": 0,
                        "target_caption": "first action",
                        "event_anaphora": True,
                        "depends_on_segment_ids": [999],
                        "keep_body_parts": ["left_hand"],
                        "action_switch_times": [1.25],
                        "no_action_switch": False,
                    },
                    {
                        "index": 1,
                        "target_caption": "second action",
                        "event_anaphora": False,
                        "depends_on_segment_ids": [999],
                        "keep_body_parts": [],
                        "action_switch_times": [],
                        "no_action_switch": True,
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            annotation_app.DATA["modified"][0]["depends_on_segment_ids"],
            [7, 8],
        )
        self.assertEqual(
            annotation_app.DATA["modified"][1]["depends_on_segment_ids"],
            [9],
        )

    def test_episode_update_is_atomic_when_caption_is_invalid(self):
        response = self.client.post(
            "/api/update_episode",
            json={
                "segments": [
                    {
                        "index": 0,
                        "target_caption": "   ",
                        "action_switch_times": [1.5],
                        "no_action_switch": False,
                    },
                    {
                        "index": 1,
                        "target_caption": "valid caption",
                        "action_switch_times": [],
                        "no_action_switch": True,
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])
        self.assertEqual(annotation_app.DATA["modified"], {})

    def test_single_update_saves_caption(self):
        response = self.client.post(
            "/api/update",
            json={
                "index": 0,
                "target_caption": "single edited caption",
                "action_switch_times": [],
                "no_action_switch": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            annotation_app.DATA["modified"][0]["target_caption"],
            "single edited caption",
        )

    def test_completed_episode_exports_valid_jsonl_in_original_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = annotation_app.timestamped_output_path(
                "/data/input.jsonl",
                temporary_directory,
                datetime(2026, 7, 14, 13, 0, 0, 1),
            )
            annotation_app.DATA["output_path"] = str(output_path)
            expected_output_path = annotation_app.range_output_path(
                output_path, 1, 1, 1
            )

            update_response = self.client.post(
                "/api/update_episode",
                json={
                    "segments": [
                        {
                            "index": 0,
                            "target_caption": "exported first caption",
                            "action_switch_times": [3.2, 1.1, 3.2],
                            "no_action_switch": False,
                        },
                        {
                            "index": 1,
                            "target_caption": "second action",
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
            self.assertEqual(save_result["total_episodes"], 1)
            self.assertEqual(save_result["modified_count"], 2)
            self.assertEqual(save_result["action_switch_annotated_count"], 2)
            self.assertEqual(save_result["episode_range"], {"start": 1, "end": 1})
            self.assertEqual(save_result["saved_to"], str(expected_output_path))

            exported_entries = [
                json.loads(line)
                for line in expected_output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(exported_entries), 2)
            self.assertEqual(
                [entry["target_segment_id"] for entry in exported_entries],
                [1, 2],
            )
            self.assertEqual(
                [entry["target_caption"] for entry in exported_entries],
                ["exported first caption", "second action"],
            )
            self.assertEqual(exported_entries[0]["action_switch_times"], [1.1, 3.2])
            self.assertFalse(exported_entries[0]["no_action_switch"])
            self.assertEqual(exported_entries[1]["action_switch_times"], [])
            self.assertTrue(exported_entries[1]["no_action_switch"])
            self.assertTrue(all(entry["annotated"] for entry in exported_entries))
            self.assertEqual(list(Path(temporary_directory).glob("*.tmp")), [])

    def test_save_exports_only_requested_episode_range(self):
        self.configure_episode_count(3)
        for index, entry in enumerate(annotation_app.DATA["entries"]):
            annotation_app.DATA["modified"][index] = {
                **entry,
                "target_caption": f"edited {entry['target_caption']}",
                "action_switch_times": [],
                "no_action_switch": True,
                "annotated": True,
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            base_output = Path(temporary_directory) / "annotations.jsonl"
            annotation_app.DATA["output_path"] = str(base_output)

            second_range_response = self.client.post(
                "/api/save",
                json={"range_start": 2, "range_end": 3},
            )
            first_range_response = self.client.post(
                "/api/save",
                json={"range_start": 1, "range_end": 1},
            )

            self.assertEqual(second_range_response.status_code, 200)
            second_result = second_range_response.get_json()
            self.assertEqual(second_result["episode_range"], {"start": 2, "end": 3})
            self.assertEqual(second_result["total_episodes"], 2)
            self.assertEqual(second_result["total_entries"], 2)
            self.assertEqual(second_result["modified_count"], 2)
            self.assertTrue(second_result["complete"])

            second_output = Path(second_result["saved_to"])
            first_output = Path(first_range_response.get_json()["saved_to"])
            second_entries = [
                json.loads(line)
                for line in second_output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [entry["episode_id"] for entry in second_entries],
                ["episode-002", "episode-003"],
            )
            self.assertEqual(second_output.name, "annotations_episodes_002-003.jsonl")
            self.assertEqual(first_output.name, "annotations_episodes_001-001.jsonl")
            self.assertTrue(second_output.exists())
            self.assertTrue(first_output.exists())
            self.assertFalse(base_output.exists())

    def test_save_rejects_invalid_episode_range_without_writing(self):
        self.configure_episode_count(3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_app.DATA["output_path"] = str(
                Path(temporary_directory) / "annotations.jsonl"
            )

            response = self.client.post(
                "/api/save",
                json={"range_start": 2, "range_end": 4},
            )

            self.assertEqual(response.status_code, 400)
            self.assertFalse(response.get_json()["ok"])
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

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
