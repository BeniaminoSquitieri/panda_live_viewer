import unittest

from bt_planning.scene_facts import build_scene_facts_stub
from bt_planning.scene_facts import build_object_pose_fact


class SceneFactsTests(unittest.TestCase):
    def test_build_scene_facts_stub(self):
        facts = build_scene_facts_stub(
            front_available=True,
            wrist_available=False,
            timestamps={"front": "123.4"},
        )

        self.assertEqual(facts["schema_version"], 1)
        self.assertEqual(facts["camera_frames"]["front"], {"available": True, "stamp": "123.4"})
        self.assertEqual(facts["camera_frames"]["wrist"], {"available": False})
        self.assertEqual(facts["facts"], {})
        self.assertEqual(facts["warnings"], [])

    def test_scene_facts_accepts_pose_facts_and_camera_details(self):
        pose_fact = build_object_pose_fact(
            name="toast",
            present=True,
            frame_id="base_link",
            stamp=1.25,
            translation={"x": 0.1, "y": 0.2, "z": 0.3},
            quaternion_xyzw={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            pose_confidence=0.9,
            covariance=[0.0] * 36,
            seg_score=0.8,
            warnings=["orientation_unestimated_centroid_fallback"],
        )
        facts = build_scene_facts_stub(
            front_available=True,
            wrist_available=True,
            timestamps={"front": 1.25, "wrist": 1.2},
            camera_details={
                "front": {
                    "frame_id": "front_camera",
                    "depth_available": True,
                    "camera_info_available": True,
                    "tf_available": True,
                }
            },
            facts={"toast": pose_fact},
            warnings=["front:orientation_unestimated_centroid_fallback"],
        )

        self.assertEqual(facts["camera_frames"]["front"]["frame_id"], "front_camera")
        self.assertTrue(facts["camera_frames"]["front"]["depth_available"])
        self.assertEqual(facts["facts"]["toast"]["frame_id"], "base_link")
        self.assertEqual(facts["facts"]["toast"]["pose_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
