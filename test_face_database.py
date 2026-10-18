import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from face_scanner import (
    COSINE_DISTANCE_THRESHOLD,
    CameraWindow,
    FaceDatabase,
    FaceDetection,
    FaceEngine,
    MIN_FACE_SIZE,
    assess_face_quality,
)


class FaceDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "faces.pkl"
        self.database = FaceDatabase(self.database_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_user(self, username: str = "user-1") -> None:
        self.database.add_user(
            username,
            {"name": "Test User", "username": username},
            [np.array([1.0, 0.0], dtype=np.float32)],
        )

    def test_add_user_persists_and_lists_display_name(self) -> None:
        self.add_user()

        reopened = FaceDatabase(self.database_path)
        self.assertEqual(reopened.list_users(), [("user-1", "Test User")])
        self.assertTrue(reopened.contains("user-1"))

    def test_add_user_can_store_both_face_representations(self) -> None:
        upper = np.array([1.0, 0.0], dtype=np.float32)
        full = np.array([0.0, 1.0], dtype=np.float32)
        self.database.add_user(
            "user-1",
            {"name": "Test User", "username": "user-1"},
            [upper],
            [full],
        )

        self.assertEqual(self.database.count_users_with_full_embeddings(), 1)
        self.assertIsNone(self.database.find_match(upper, "full"))
        self.assertEqual(self.database.find_match(full, "full")[0], "user-1")

    def test_legacy_user_remains_matchable_with_upper_representation(self) -> None:
        self.add_user()

        self.assertEqual(
            self.database.find_match(np.array([1.0, 0.0], dtype=np.float32), "upper")[
                0
            ],
            "user-1",
        )
        self.assertIsNone(
            self.database.find_match(np.array([1.0, 0.0], dtype=np.float32), "full")
        )

    def test_update_user_rolls_back_when_persistence_fails(self) -> None:
        self.add_user()
        replacement = np.array([0.0, 1.0], dtype=np.float32)

        with patch.object(self.database, "_save_locked", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.database.update_user(
                    "user-1",
                    {"name": "Replacement", "username": "user-1"},
                    [replacement],
                    [replacement],
                )

        self.assertEqual(self.database.get_details("user-1")["name"], "Test User")
        self.assertEqual(
            self.database.find_match(np.array([1.0, 0.0], dtype=np.float32))[0],
            "user-1",
        )

    def test_find_match_returns_best_match_within_threshold(self) -> None:
        self.add_user()

        match = self.database.find_match(np.array([1.0, 0.01], dtype=np.float32))

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match[0], "user-1")
        self.assertLessEqual(match[1], COSINE_DISTANCE_THRESHOLD)

    def test_find_match_rejects_impostor_within_previous_threshold(self) -> None:
        self.add_user()
        impostor = np.array([0.5, np.sqrt(0.75)], dtype=np.float32)

        self.assertIsNone(self.database.find_match(impostor))

    def test_find_match_requires_consistency_across_enrollment_samples(self) -> None:
        self.database.add_user(
            "user-1",
            {"name": "Test User", "username": "user-1"},
            [
                np.array([0.7, np.sqrt(0.51)], dtype=np.float32),
                np.array([0.3, np.sqrt(0.91)], dtype=np.float32),
            ],
        )

        self.assertIsNone(
            self.database.find_match(np.array([1.0, 0.0], dtype=np.float32))
        )

    def test_find_match_rejects_distant_embedding(self) -> None:
        self.add_user()

        self.assertIsNone(
            self.database.find_match(np.array([0.0, 1.0], dtype=np.float32))
        )

    def test_find_match_rejects_zero_embedding(self) -> None:
        self.add_user()

        self.assertIsNone(
            self.database.find_match(np.array([0.0, 0.0], dtype=np.float32))
        )

    def test_delete_user_removes_and_persists_record(self) -> None:
        self.add_user()

        self.database.delete_user("user-1")

        self.assertEqual(self.database.list_users(), [])
        self.assertFalse(FaceDatabase(self.database_path).contains("user-1"))

    def test_delete_missing_user_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.database.delete_user("missing")

    def test_delete_rolls_back_when_persistence_fails(self) -> None:
        self.add_user()

        with patch.object(self.database, "_save_locked", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.database.delete_user("user-1")

        self.assertTrue(self.database.contains("user-1"))

    def test_add_rolls_back_when_persistence_fails(self) -> None:
        with patch.object(self.database, "_save_locked", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.add_user()

        self.assertFalse(self.database.contains("user-1"))

    def test_face_quality_rejects_small_face(self) -> None:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)

        self.assertIsNotNone(
            assess_face_quality(frame, (10, 10, MIN_FACE_SIZE - 1, MIN_FACE_SIZE))
        )

    def test_face_quality_rejects_blurred_face(self) -> None:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)

        self.assertIn(
            "غير واضحة",
            assess_face_quality(frame, (10, 10, MIN_FACE_SIZE, MIN_FACE_SIZE)) or "",
        )

    def test_face_quality_accepts_clear_well_lit_face(self) -> None:
        random = np.random.default_rng(0)
        frame = random.integers(40, 216, size=(240, 320, 3), dtype=np.uint8)

        self.assertIsNone(
            assess_face_quality(frame, (10, 10, MIN_FACE_SIZE, MIN_FACE_SIZE))
        )

    def test_digital_zoom_crops_around_detected_face(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = FaceDetection(box=(500, 200, 80, 80))

        zoomed = CameraWindow._zoom_preview(frame, [detection], 2)

        self.assertEqual(zoomed.shape[:2], (240, 320))

    def test_compare_mode_matches_upper_and_full_representations(self) -> None:
        self.database.add_user(
            "user-1",
            {"name": "Test User", "username": "user-1"},
            [np.array([1.0, 0.0], dtype=np.float32)],
            [np.array([0.0, 1.0], dtype=np.float32)],
        )
        engine = FaceEngine.__new__(FaceEngine)
        engine.deepface = SimpleNamespace(
            represent=Mock(
                side_effect=[
                    [{"embedding": [1.0, 0.0]}],
                    [{"embedding": [0.0, 1.0]}],
                ]
            )
        )
        engine._extract_faces = Mock(
            return_value=[
                {
                    "face": np.zeros((112, 112, 3), dtype=np.uint8),
                    "facial_area": {"x": 10, "y": 10, "w": 100, "h": 100},
                }
            ]
        )

        with patch("face_scanner.assess_face_quality", return_value=None):
            detections = engine.analyze_frame(
                np.zeros((200, 200, 3), dtype=np.uint8),
                self.database,
                representation="compare",
            )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].username, "user-1")
        self.assertEqual(detections[0].alternate_username, "user-1")
        self.assertEqual(engine.deepface.represent.call_count, 2)


if __name__ == "__main__":
    unittest.main()
