import unittest
import json
from unittest.mock import patch
from app import app


class TestErrorCases(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test for invalid JSON request
    def test_invalid_json(self):
        response = self.app.post('/generate',
                                 data="Not a JSON",
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid input format", response.data)

    # Test for missing 'prompt' field
    def test_missing_prompt(self):
        payload = {}  # No 'prompt' field
        response = self.app.post('/generate',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"'prompt' field is required", response.data)

    # Mocking an internal server error (like a model loading failure)
    @patch('app.stable_diffusion_pipe')
    def test_model_error(self, mock_model):
        # Simulate a model error
        mock_model.side_effect = Exception('Model failed to load')

        payload = {
            "prompt": "a mom holding a baby"
        }
        response = self.app.post('/generate',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 500)
        self.assertIn(b"An error occurred", response.data)


if __name__ == '__main__':
    unittest.main()
