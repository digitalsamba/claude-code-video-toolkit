import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review import ReviewError, ReviewStore, StaleReviewError, make_server


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'public').mkdir()
        (self.root / 'public/preview.png').write_bytes(b'picture')
        self.project = {'name': '<unsafe> name', 'template': 'custom', 'scenes': [
            {'id': 'intro', 'type': 'title', 'durationSeconds': 4, 'narration': 'First words',
             'visual': {'type': 'screenshot', 'asset': 'preview.png'}}]}
        self.write()
        self.store = ReviewStore(self.root)

    def write(self):
        (self.root / 'project.json').write_text(json.dumps(self.project), encoding='utf-8')

    def action(self, action, **extra):
        scene = self.store.snapshot()['scenes'][0]
        return self.store.mutate({'sceneId': 'intro', 'fingerprint': scene['fingerprint'], 'action': action, **extra})

    def test_read_does_not_write_and_media_is_allowlisted(self):
        scene = self.store.snapshot()['scenes'][0]
        self.assertEqual(scene['reviewStatus'], 'pending')
        self.assertEqual(scene['narration'], 'First words')
        self.assertIsNotNone(self.store.media_path(scene['preview']['url'].split('/')[-1]))
        self.assertIsNone(self.store.media_path('0' * 24))
        self.assertFalse((self.root / '.video-toolkit').exists())

    def test_approval_and_comments_survive_restart_but_changes_expire_approval(self):
        self.action('comment', text='<script>alert(1)</script>', timeSeconds=1.25)
        self.action('approve')
        self.assertEqual(ReviewStore(self.root).snapshot()['scenes'][0]['reviewStatus'], 'approved')
        self.project['scenes'][0]['narration'] = 'Edited'
        self.write()
        scene = self.store.snapshot()['scenes'][0]
        self.assertEqual(scene['reviewStatus'], 'stale')
        self.assertEqual(scene['comments'][0]['timeSeconds'], 1.25)
        self.action('approve')
        (self.root / 'public/preview.png').write_bytes(b'edited picture')
        self.assertEqual(self.store.snapshot()['scenes'][0]['reviewStatus'], 'stale')

    def test_stale_client_cannot_approve_new_content(self):
        scene = self.store.snapshot()['scenes'][0]
        self.project['scenes'][0]['narration'] = 'changed'
        self.write()
        with self.assertRaises(StaleReviewError):
            self.store.mutate({'sceneId': 'intro', 'fingerprint': scene['fingerprint'], 'action': 'approve'})

    def test_missing_media_and_bad_comments(self):
        (self.root / 'public/preview.png').unlink()
        self.assertIn('Missing asset', self.store.snapshot()['scenes'][0]['issues'][0])
        for kwargs in [{'text': ''}, {'text': 'hello', 'timeSeconds': -1}, {'text': 'a' * 4001}]:
            with self.assertRaises(ReviewError):
                self.action('comment', **kwargs)

    def test_build_qc_report_is_displayed(self):
        report = self.root / '.video-toolkit/reports/qc-after.json'
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({'findings': [{'level': 'warning', 'message': 'Black frame', 'startSeconds': 2}]}))
        self.assertEqual(self.store.snapshot()['warnings'][0]['timeSeconds'], 2)

    def test_http_mutations_require_token_and_media_ranges_work(self):
        server = make_server(self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(3)
        self.addCleanup(cleanup)
        def request(method, path, payload=None, headers=None):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            body = json.dumps(payload) if payload is not None else None
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            connection.close()
            return result
        status, headers, raw = request('GET', '/api/project')
        self.assertEqual(status, 200)
        self.assertIn("script-src 'self'", headers['Content-Security-Policy'])
        snapshot = json.loads(raw)
        scene = snapshot['scenes'][0]
        body = {'sceneId': 'intro', 'fingerprint': scene['fingerprint'], 'action': 'approve'}
        self.assertEqual(request('POST', '/api/review', body, {'Content-Type': 'application/json'})[0], 403)
        mutation_headers = {'Content-Type': 'application/json', 'X-Review-Token': snapshot['token']}
        self.assertEqual(request('POST', '/api/review', body, mutation_headers)[0], 200)
        mutation_headers['Origin'] = 'https://untrusted.invalid'
        self.assertEqual(request('POST', '/api/review', body, mutation_headers)[0], 403)
        self.assertEqual(request('GET', '/api/project', headers={'Host': 'untrusted.invalid'})[0], 403)
        status, headers, raw = request('GET', scene['preview']['url'], headers={'Range': 'bytes=1-3'})
        self.assertEqual(status, 206)
        self.assertEqual(raw, b'ict')
        self.assertEqual(request('GET', '/.env')[0], 404)


if __name__ == '__main__':
    unittest.main()
