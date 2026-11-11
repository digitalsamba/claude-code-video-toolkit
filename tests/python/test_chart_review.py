import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import data_video


class ChartReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def report(self, periods, *, kind='line', names=('A',), **options):
        rows = [{'period': period, 'series': name, 'value': (i + 1) * (j + 1)}
                for j, name in enumerate(names) for i, period in enumerate(periods)]
        (self.folder / 'data.json').write_text(json.dumps(rows), encoding='utf-8')
        spec = self.folder / 'chart.json'
        spec.write_text(json.dumps({'version': 1, 'type': kind, 'dataset': 'data.json',
                                    'width': 640, 'height': 360, 'fps': 20, 'durationSeconds': 2,
                                    **options}), encoding='utf-8')
        return data_video.prepare(spec, brand='digital-samba')

    def renderer(self, report):
        figure, frame_at = data_video.chart_renderer(report)
        self.addCleanup(figure.clear)
        return figure, frame_at

    def test_line_spacing_and_reveal_use_real_dates_and_month_labels(self):
        report = self.report(['2026-01-01', '2026-01-03', '2026-03-01'])
        figure, frame_at = self.renderer(report)
        frame_at(2)
        xs = np.array(figure.axes[0].lines[0].get_xdata(), dtype=float)
        np.testing.assert_allclose(np.diff(xs[:3]), [2, 57])
        labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
        self.assertTrue(labels)
        self.assertTrue(all(re.fullmatch(r'[A-Z][a-z]{2} 2026', label) for label in labels), labels)
        frame_at(.5)
        head = figure.axes[0].lines[0].get_xdata()[-1]
        self.assertGreater(head - xs[0], 2)
        self.assertLess(head, xs[2])

    def test_date_series_sorts_chronologically_with_matching_values(self):
        report = self.report(['2026-03-01', '2026-01-01', '2026-02-01'])
        figure, frame_at = self.renderer(report)
        frame_at(2)
        line = figure.axes[0].lines[0]
        np.testing.assert_allclose(np.diff(line.get_xdata()[:3]), [31, 28])
        np.testing.assert_allclose(line.get_ydata()[:3], [2, 3, 1])

    def test_month_periods_use_calendar_spacing_and_categories_stay_categorical(self):
        for periods, expected in [(['2026-01', '2026-03'], 59), (['Jan 2026', 'Mar 2026'], 59)]:
            figure, frame_at = self.renderer(self.report(periods))
            frame_at(2)
            xs = figure.axes[0].lines[0].get_xdata()
            self.assertAlmostEqual(xs[1] - xs[0], expected)
        figure, frame_at = self.renderer(self.report(['Alpha', 'Beta', 'Gamma']))
        frame_at(2)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata()[:3], [0, 1, 2])
        self.assertEqual([tick.get_text() for tick in figure.axes[0].get_xticklabels()], ['Alpha', 'Beta', 'Gamma'])

    def test_single_line_has_no_legend_and_secondary_lines_share_one_neutral(self):
        report = self.report(['Jan', 'Feb'])
        figure, _ = self.renderer(report)
        self.assertIsNone(figure.axes[0].get_legend())
        self.assertEqual(colors.to_rgba(figure.axes[0].lines[0].get_color()), colors.to_rgba('#F06859'))
        report = self.report(['Jan', 'Feb'], names=('A', 'B', 'C', 'D'))
        figure, _ = self.renderer(report)
        rendered = [colors.to_rgba(line.get_color()) for line in figure.axes[0].lines]
        self.assertEqual(rendered[0], colors.to_rgba('#F06859'))
        self.assertEqual(rendered[1:], [colors.to_rgba(report['theme']['muted'])] * 3)
        self.assertIsNotNone(figure.axes[0].get_legend())

    def test_bars_share_primary_and_period_text_is_visible_and_changes(self):
        report = self.report(['January', 'February'], kind='bar', names=('A', 'B', 'C'))
        figure, frame_at = self.renderer(report)
        expected = colors.to_rgba('#F06859')
        self.assertEqual([bar.get_facecolor() for bar in figure.axes[0].patches], [expected] * 3)
        first, middle, last = [frame_at(t) for t in (0, .5, 2)]
        # The period occupies the otherwise empty band between headline and plot.
        first, middle, last = [frame[70:86, 20:620] for frame in (first, middle, last)]
        self.assertGreater(np.abs(first.astype(float) - 255).sum(), 0)
        self.assertGreater(np.abs(first.astype(float) - last).sum(), 0)
        self.assertGreater(np.abs(middle.astype(float) - last).sum(), 0)

    def test_headline_starts_hidden_and_fades_at_authored_time(self):
        report = self.report(['Jan', 'Feb'], headline='718 stars', headlineStartSeconds=.4, headlineFadeSeconds=.4)
        _, frame_at = self.renderer(report)
        before, halfway, visible = [frame_at(t)[40:69, 20:620].astype(float) for t in (.2, .6, 1)]
        self.assertEqual(np.abs(before - 255).sum(), 0)
        partial, full = np.abs(halfway - 255).sum(), np.abs(visible - 255).sum()
        self.assertGreater(partial, 0)
        self.assertGreater(full, partial)
        self.assertAlmostEqual(partial / full, .5, delta=.05)

    def test_invalid_headline_timing_is_rejected(self):
        for options in ({'headlineStartSeconds': -1}, {'headlineStartSeconds': float('nan')},
                        {'headlineStartSeconds': 3}, {'headlineFadeSeconds': -1},
                        {'headlineStartSeconds': 1.8, 'headlineFadeSeconds': .5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.report(['Jan', 'Feb'], headline='718 stars', **options)

    def test_default_generic_font_is_not_reported_as_missing(self):
        theme = data_video.brand_theme('default')
        self.assertFalse(theme['fontFallback'])
        self.assertTrue(Path(theme['fontPath']).is_file())
        brand = json.loads((ROOT / 'brands/default/brand.json').read_text(encoding='utf-8'))
        for family, expected in [('sans-serif', False), ('Missing Toolkit Review Font, sans-serif', True)]:
            configured = copy.deepcopy(brand)
            configured['fonts']['primary'] = family
            with patch.object(data_video, 'load_json', return_value=configured):
                self.assertEqual(data_video.brand_theme('default')['fontFallback'], expected)

    def test_mixed_and_duplicate_date_periods_are_rejected(self):
        for periods in (['2026-01-01', 'not a date'], ['2026-01', 'Jan 2026']):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                self.report(periods)

    def test_short_date_range_has_a_readable_month_tick(self):
        figure, frame_at = self.renderer(self.report(['2026-01-20', '2026-01-22']))
        frame_at(2)
        self.assertEqual([t.get_text() for t in figure.axes[0].get_xticklabels()], ['Jan 2026'])

    def test_zero_duration_headline_fade_reveals_at_exact_time(self):
        report = self.report(['Jan', 'Feb'], headline='718 stars', headlineStartSeconds=.4, headlineFadeSeconds=0)
        _, frame_at = self.renderer(report)
        before = frame_at(.3)[40:69, 20:620].astype(float)
        after = frame_at(.4)[40:69, 20:620].astype(float)
        self.assertEqual(np.abs(before - 255).sum(), 0)
        self.assertGreater(np.abs(after - 255).sum(), 0)

    def test_bundled_spec_restores_author_headline_and_timing(self):
        report = data_video.prepare(ROOT / 'examples/data-viz-chart/chart.json', brand='digital-samba')
        self.assertEqual(report['headline'], '718 stars')
        self.assertEqual(report['headlineStartSeconds'], 10)
        self.assertEqual(report['headlineFadeSeconds'], .5)
