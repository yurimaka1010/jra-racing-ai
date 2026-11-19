"""Small synthetic input joins; labels and money are independent."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import pandas as pd
from jra_racing_ai.multiwager import preparation as m, models
from jra_racing_ai.ml.safety import MLFailure


def frame():
    return pd.DataFrame([dict.fromkeys(models.base_spec()['features'], None) | {
        'entry_key':'2025030105010101'+h,'date':'2025-03-01', 'winner_label':int(i==1),
        'finish_position':i,'actual_starters':3} for i,h in enumerate(['01','02','03'],1)])


class TrainingInputTests(unittest.TestCase):
    def test_all_missing_optional_features_do_not_remove_confirmed_label_rows(self):
        f=frame();race='2025030105010101'
        out,status=m.attach_labels(f,{race:2},{race:{'place':{'target_status':'confirmed','official_targets':['01','02']}}})
        self.assertEqual(len(out),3);self.assertEqual(out.official_place_label.tolist(),[1,1,0])
        self.assertEqual(status['excluded_reasons'],{});self.assertFalse(status['optional_features_used_for_filtering'])

    def test_headcount_is_never_a_substitute_for_missing_asof_pay_key(self):
        with self.assertRaises(MLFailure):m.attach_labels(frame(),{}, {'2025030105010101':{'place':{'target_status':'confirmed','official_targets':['01','02']}}})

    def test_unverified_hr_set_never_gets_verified_label_status(self):
        with self.assertRaises(MLFailure):m.attach_labels(frame(),{'2025030105010101':2},{'2025030105010101':{'place':{'target_status':'pending'}}})

    def test_input_hash_checked_before_parsing_results(self):
        with patch.object(m.lab,'check',side_effect=MLFailure('artifact_invalid')),patch.object(m,'read_frame') as reader:
            with self.assertRaises(MLFailure):m.verified_part(Path('x'),'a'*64,['2023-01-01','2024-08-31'])
            reader.assert_not_called()

    def test_sealed_identifier_stops_before_full_label_file_is_read(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'bad.csv';path.write_text('entry_key,date\n202601010501010101,2026-01-01\n',encoding='utf-8')
            with patch.object(m.lab,'check'),patch.object(m,'read_frame') as reader:
                with self.assertRaises(MLFailure):m.verified_part(path,'a'*64,['2023-01-01','2024-08-31'])
                reader.assert_not_called()

