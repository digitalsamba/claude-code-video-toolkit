import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import reference_capture

JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 64


class ReferenceCaptureServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), reference_capture.make_handler(self.out))
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def post(self, name, body, headers=None):
        req = urllib.request.Request(f'{self.base}/save?name={name}', data=body, headers=headers or {}, method='POST')
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.read()

    def test_page_lists_every_pose(self):
        page = self.get('/').decode()
        for pose in reference_capture.POSES:
            self.assertIn(pose['name'], page)

    def test_a_saved_pose_is_written_listed_and_served(self):
        self.assertEqual(json.loads(self.get('/list')), [])
        self.assertEqual(self.post('front', JPEG), 200)
        self.assertEqual((self.out / 'front.jpg').read_bytes(), JPEG)
        self.assertEqual(json.loads(self.get('/list')), ['front'])
        self.assertEqual(self.get('/img/front.jpg'), JPEG)
        self.assertEqual(list(self.out.glob('*.part')), [])

    def test_only_known_poses_are_saved(self):
        self.assertEqual(self.post('passwd', JPEG), 400)
        self.assertEqual(self.post('../front', JPEG), 400)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_a_file_that_is_not_a_jpeg_is_refused(self):
        self.assertEqual(self.post('front', b'<html>'), 400)
        self.assertFalse((self.out / 'front.jpg').exists())

    def test_another_site_cannot_save(self):
        self.assertEqual(self.post('front', JPEG, {'Origin': 'https://example.com'}), 403)
        self.assertEqual(self.post('front', JPEG, {'Origin': 'http://localhost:8765'}), 200)


if __name__ == '__main__':
    unittest.main()
