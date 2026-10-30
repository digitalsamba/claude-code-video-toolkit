import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from asset_history import AssetHistoryError, list_assets, project_path, record_asset, restore_asset


class AssetHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.asset = self.root / 'public' / 'title.png'
        self.asset.parent.mkdir()
        self.asset.write_bytes(b'first image')

    def test_restore_old_bytes_and_keep_both_versions(self):
        first = record_asset(self.root, self.asset, metadata={'prompt': 'Blue sea', 'seed': 4},
                             kind='image', scene_id='intro')
        self.asset.write_bytes(b'second image')
        record_asset(self.root, self.asset, metadata={'prompt': 'Red sky'}, kind='image')
        with self.assertRaisesRegex(AssetHistoryError, 'already exists'):
            restore_asset(self.root, first['id'])
        restore_asset(self.root, first['id'], force=True)
        self.assertEqual(self.asset.read_bytes(), b'first image')
        matches = list_assets(self.root, query='blue SEA', scene_id='intro', kind='image')
        self.assertEqual([item['id'] for item in matches], [first['id']])
        self.assertEqual(len(list_assets(self.root)), 2)

    def test_duplicate_content_shares_blob_but_preserves_provenance(self):
        one = record_asset(self.root, self.asset, metadata={'seed': 1})
        two = record_asset(self.root, self.asset, metadata={'seed': 2})
        self.assertEqual(one['sha256'], two['sha256'])
        self.assertNotEqual(one['id'], two['id'])
        self.assertEqual(len(list((self.root / '.video-toolkit/assets/objects').iterdir())), 1)

    def test_concurrent_records_do_not_lose_versions(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda seed: record_asset(self.root, self.asset,
                                metadata={'seed': seed}), range(12)))
        self.assertEqual(len(list_assets(self.root)), 12)
        self.assertEqual(len({item['id'] for item in records}), 12)

    def test_missing_history_is_read_only(self):
        self.assertEqual(list_assets(self.root), [])
        self.assertFalse((self.root / '.video-toolkit').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows path aliases')
    def test_extended_windows_prefix_does_not_reject_a_project_path(self):
        root = self.root.resolve()
        extended = Path('\\\\?\\' + str(root / 'public/title.png'))
        with patch.object(Path, 'resolve', side_effect=[root, extended]):
            self.assertEqual(project_path(root, Path('public/title.png')), root / 'public/title.png')

    def test_paths_and_record_ids_stay_in_project(self):
        with self.assertRaisesRegex(AssetHistoryError, 'inside the project'):
            record_asset(self.root, Path('../outside.png'))
        record = record_asset(self.root, self.asset)
        with self.assertRaisesRegex(AssetHistoryError, 'inside the project'):
            restore_asset(self.root, record['id'], output=Path('../outside.png'))
        with self.assertRaisesRegex(AssetHistoryError, '32 lowercase'):
            restore_asset(self.root, '../other')
        with self.assertRaisesRegex(AssetHistoryError, 'history store itself'):
            record_asset(self.root, self.root / '.video-toolkit/assets/objects' / record['sha256'])

    def test_corrupt_snapshots_are_not_restored(self):
        record = record_asset(self.root, self.asset)
        blob = self.root / '.video-toolkit/assets/objects' / record['sha256']
        blob.write_bytes(b'corruption')
        with self.assertRaisesRegex(AssetHistoryError, 'corrupt'):
            restore_asset(self.root, record['id'], output=Path('restored.png'))
        self.assertFalse((self.root / 'restored.png').exists())

    def test_invalid_metadata_does_not_create_history(self):
        for metadata in ([], {'temperature': float('nan')}):
            with self.assertRaises(AssetHistoryError):
                record_asset(self.root, self.asset, metadata=metadata)
        self.assertFalse((self.root / '.video-toolkit').exists())

    def test_cli_record_search_and_restore(self):
        tool = Path(__file__).resolve().parents[1] / 'tools/asset_history.py'
        command = [sys.executable, str(tool), '--project', str(self.root), '--json']
        result = subprocess.run([*command, 'record', 'public/title.png'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        result = subprocess.run([*command, 'list', '--query', 'title.png'],
                                capture_output=True, text=True)
        self.assertEqual(len(json.loads(result.stdout)), 1)
        result = subprocess.run([*command, 'restore', record['id'], '--output', 'restored.png'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'restored.png').read_bytes(), b'first image')


if __name__ == '__main__':
    unittest.main()
