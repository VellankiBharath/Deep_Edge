import unittest
import json
from app import app  # Assuming app.py contains your Flask app

class TestAPI(unittest.TestCase):

    # Setup the Flask test client
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test the root endpoint
    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome to the Deep Edge API!", response.data)

    # Test the /generate endpoint with a valid prompt
    def test_generate_image(self):
        payload = {
            "prompt": "a beautiful landscape with mountains"
        }
        response = self.app.post('/generate',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Check if response contains required fields
        response_data = json.loads(response.data)
        self.assertIn('request_id', response_data)
        self.assertIn('generated_image', response_data)
        self.assertIn('clip_analysis', response_data)
        self.assertIn('basic_segmentation', response_data)

    # Test the /generate endpoint with an invalid prompt
    def test_generate_invalid_input(self):
        payload = {
            "invalid_field": "no prompt"
        }
        response = self.app.post('/generate',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)  # Expecting Bad Request (400)

    # Test the /data endpoint
    def test_get_data(self):
        response = self.app.get('/data')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
