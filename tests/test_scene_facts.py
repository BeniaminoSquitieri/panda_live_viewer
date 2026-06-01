import unittest

from bt_planning.scene_facts import build_scene_facts_stub


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


if __name__ == "__main__":
    unittest.main()
