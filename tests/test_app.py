import unittest
from collections import OrderedDict
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


if __name__ == "__main__":
    unittest.main()
