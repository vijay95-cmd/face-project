import unittest
import os
import tempfile
import shutil
from config import config

class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Mock config for testing
        self.original_config = config.copy()
        config['paths']['dataset_dir'] = os.path.join(self.test_dir, 'dataset')
        config['paths']['database_file'] = os.path.join(self.test_dir, 'test.db')

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        config.update(self.original_config)

    def test_blur_detection(self):
        from dataset import is_blurry
        import cv2
        import numpy as np

        # Create a sharp test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (25, 25), (75, 75), (255, 255, 255), -1)
        blurry, sharpness = is_blurry(test_img)
        self.assertIsInstance(blurry, bool)
        self.assertIsInstance(sharpness, (int, float))
        self.assertFalse(blurry)  # Sharp image should not be blurry

if __name__ == '__main__':
    unittest.main()
