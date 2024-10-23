import unittest
from io import BytesIO
from PIL import Image
from app import image_to_base64

class TestCoreFunctions(unittest.TestCase):

    def test_image_to_base64(self):
        # Create a test image in memory
        img = Image.new('RGB', (10, 10), color = 'red')
        result = image_to_base64(img)

        # Ensure the result is a string
        self.assertIsInstance(result, str)
        # Check if the result is a valid base64 string
        self.assertTrue(result.startswith('iVBOR'))

    # You can add more core function tests here

if __name__ == '__main__':
    unittest.main()
