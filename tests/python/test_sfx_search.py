import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import sfx_search


def record(duration, license='cc0', version='1.0'):
    return {
        'id': 'abc', 'title': 'Whoosh', 'creator': 'someone', 'duration': duration,
        'license': license, 'license_version': version, 'license_url': 'https://cc',
        'attribution': 'Whoosh by someone', 'url': 'https://cdn/whoosh.mp3',
        'foreign_landing_url': 'https://freesound.org/s/1/',
    }


def response(status=200, json_body=None, headers=None, content=b''):
    fake = mock.Mock(status_code=status, ok=status < 400, headers=headers or {}, content=content)
    fake.json.return_value = json_body
    return fake


class SearchTests(unittest.TestCase):
    def test_milliseconds_become_seconds_and_long_or_unknown_sounds_drop(self):
        body = {'results': [record(1500), record(60000), record(None)]}
        with mock.patch.object(sfx_search.requests, 'get', return_value=response(json_body=body)) as get:
            results = sfx_search.search('whoosh', 'cc0,by', 10, 10)
        self.assertEqual([r['duration_seconds'] for r in results], [1.5])
        self.assertEqual(results[0]['license'], 'cc0 1.0')
        params = get.call_args.kwargs['params']
        self.assertEqual(params['source'], 'freesound')
        self.assertNotIn('category', params)

    def test_short_rate_limit_is_retried_and_long_one_is_reported(self):
        limited = response(429, headers={'Retry-After': '1'})
        with mock.patch.object(sfx_search.requests, 'get', side_effect=[limited, response()]) as get, \
                mock.patch.object(sfx_search.time, 'sleep'):
            sfx_search.get('https://api')
        self.assertEqual(get.call_count, 2)

        limited = response(429, headers={'Retry-After': '3600'})
        with mock.patch.object(sfx_search.requests, 'get', return_value=limited):
            with self.assertRaises(sfx_search.OpenverseError):
                sfx_search.get('https://api')


class DownloadTests(unittest.TestCase):
    def test_writes_audio_and_credit(self):
        audio = response(headers={'Content-Type': 'audio/mpeg'}, content=b'mp3')
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(sfx_search.requests, 'get', return_value=audio):
            output = Path(directory) / 'sub' / 'whoosh.mp3'
            credit = sfx_search.download(sfx_search.summarize(record(1500)), output)
            self.assertEqual(output.read_bytes(), b'mp3')
            self.assertEqual(credit.name, 'whoosh.attribution.txt')
            self.assertIn('Whoosh by someone', credit.read_text(encoding='utf-8'))

    def test_refuses_a_page_served_instead_of_audio(self):
        page = response(headers={'Content-Type': 'text/html'}, content=b'<html>')
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(sfx_search.requests, 'get', return_value=page):
            output = Path(directory) / 'whoosh.mp3'
            with self.assertRaises(sfx_search.OpenverseError):
                sfx_search.download(sfx_search.summarize(record(1500)), output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
