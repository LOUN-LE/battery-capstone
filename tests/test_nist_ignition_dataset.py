"""Checks unknown labels, source timing and malfunction handling, not fire physics."""
import csv
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from simulation.nist_ignition_dataset import (
    annotation, CONTINUOUS_IDS, EXPECTED_IDS, parse_description, read_xlsx, write_csv,
)


def col_name(index):
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(65 + remainder) + value
    return value


def fixture_workbook(path, second_time=1):
    ids = sorted(EXPECTED_IDS)
    strings = "".join(f"<si><t>Exp{i}</t></si>" for i in ids) + "<si><t>na</t></si>"
    headers = "".join(f'<c r="{col_name(k+2)}1" t="s"><v>{k}</v></c>' for k in range(77))
    rows = []
    for r, time in [(3, 0), (4, second_time)]:
        cells = f'<c r="A{r}"><v>{time}</v></c>'
        for k, exp in enumerate(ids):
            col = col_name(k + 2)
            cells += (f'<c r="{col}{r}" t="s"><v>77</v></c>' if exp in {3, 17}
                      else f'<c r="{col}{r}"><v>{20+exp/100}</v></c>')
        rows.append(f'<row r="{r}">{cells}</row>')
    namespace = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", f"<sst {namespace}>{strings}</sst>")
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet {namespace}><sheetData><row r="1">{headers}</row><row r="2"/>{"".join(rows)}</sheetData></worksheet>')


class NistDatasetTests(unittest.TestCase):
    def test_unknown_figure_ids_never_become_negative(self):
        for exp in [33, 62]:
            self.assertIsNone(annotation(exp)["ignition_observed"])
            self.assertEqual(annotation(exp)["ignition_outcome"], "unknown")

    def test_nmc_negative_annotations_are_primary_figure_cases(self):
        negatives = {i for i in range(49, 83) if annotation(i)["ignition_observed"] is False}
        self.assertEqual(negatives, {65, 67, 72, 73, 74, 77, 78})
        self.assertEqual(sum(annotation(i)["ignition_observed"] is True for i in range(49, 83)), 26)

    def test_unknown_event_times_and_identity_are_not_filled(self):
        for i in EXPECTED_IDS:
            item = annotation(i)
            for key in ["ignition_time_s", "svb_time_s", "heating_start_time_s", "heating_stop_time_s", "visual_observation_end_s", "hrr_kw"]:
                self.assertIsNone(item[key])
            self.assertFalse(item["m50t_identity_verified"])

    def test_excluded_render_safe_ids_are_rejected(self):
        for exp in [18, 19, 20, 23, 27, 83]:
            with self.assertRaises(ValueError):
                annotation(exp)

    def test_description_ignores_trailing_excluded_experiments(self):
        lines = []
        for exp in sorted(EXPECTED_IDS):
            stop = "NA" if exp in CONTINUOUS_IDS else "30"
            lines += [f"Experiment #\tExp {exp}", "SOC (%)\t50", f"Intervention Time Relative to SVB (s)\t{stop}"]
        lines += ["Experiment #\tExp19\tExp20", "SOC (%)\t100\t75", "Intervention Time Relative to SVB (s)\tNA\tNA"]
        result = parse_description("\n".join(lines))
        self.assertEqual(set(result), EXPECTED_IDS)
        self.assertIsNone(result[49]["intervention_after_svb_s"])
        self.assertEqual(result[61]["intervention_after_svb_s"], 30)

    def test_source_na_and_cell_address_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.xlsx"
            fixture_workbook(path)
            result = read_xlsx(path)
            self.assertEqual(len(result), 77)
            self.assertTrue(all(r["surface_temperature_c"] is None for r in result[3]))
            self.assertEqual(result[1][0]["source_cell"], "B3")
            self.assertAlmostEqual(result[49][1]["surface_temperature_c"], 20.49)

    def test_source_non_1hz_is_rejected_without_silent_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.xlsx"
            fixture_workbook(path, second_time=2)
            with self.assertRaises(ValueError):
                read_xlsx(path)

    def test_csv_false_and_unknown_remain_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            write_csv(path, [{"label": True}, {"label": False}, {"label": None}])
            with path.open(encoding="utf-8-sig", newline="") as f:
                self.assertEqual([r["label"] for r in csv.DictReader(f)], ["true", "false", ""])


if __name__ == "__main__":
    unittest.main()
