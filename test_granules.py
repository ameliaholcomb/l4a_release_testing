import h5py
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pathlib
import random
import sys
import unittest
from typing import Dict, List

from tools.granule import Granule
from tools.granule_name import parse_granule_filename

LOCALFILES_DIR = pathlib.Path(
    "/Users/amelia/Repos/testdata/gedi_l4a_test_granules"
)
V3FILES_DIR = pathlib.Path(
    "/gpfs/data1/vclgp/armstonj/gedi_l4a_processing/gedi_l4a_r003_test"
)
V2FILES_DIR = pathlib.Path("/gpfs/data1/vclgp/data/iss_gedi/soc/2022/132")


TESTDIR = pathlib.Path(__file__).parent
DATA_DICTIONARY_DIR = TESTDIR / "data_dictionaries"


def _get_files_umd():
    v3files = list(V3FILES_DIR.iterdir())
    v3md = [parse_granule_filename(f.name) for f in v3files]
    v2f = [
        list(
            V2FILES_DIR.glob(
                f"GEDI04_A*{md.orbit}_{md.sub_orbit_granule}*_V002.h5"
            )
        )
        for md in v3md
    ]
    v2files = []
    for f in v2f:
        if len(f) == 2:
            if f[0] > f[1]:
                v2files.append(f[0])
            else:
                v2files.append(f[1])
        elif len(f) == 1:
            v2files.append(f[0])
        else:
            print("Error: unexpected number of files found\n", f)
    return list(zip(v2files, v3files))


def _get_l2a_file_umd(v3file: pathlib.Path):
    md = parse_granule_filename(v3file.name)
    l2afile = list(
        V2FILES_DIR.glob(f"GEDI02_A*{md.orbit}_{md.sub_orbit_granule}*_V003.h5")
    )
    if len(l2afile) != 1:
        raise (
            f"Expected exactly one L2A file for {v3file.name}, found {len(l2afile)}"
        )
    return l2afile[0]


def _get_files_local():
    v3files = sorted(list(LOCALFILES_DIR.glob("*V003.h5")))
    v2files = sorted(list(LOCALFILES_DIR.glob("*V002.h5")))
    return list(zip(v2files, v3files))


def _get_columns(file_path: pathlib.Path):
    columns = []
    with open(file_path, "r") as f:
        for line in f:
            columns.append(line.strip())
    return columns


def _get_grps_columns_dict(columns: List[str]) -> Dict[str, List[str]]:
    """
    Returns a dictionary of
      group_name: [column_name, ...]
    In other words, a list of all columns expected under each group.
    Only intended to be used for ONE level of hierarchy (i.e., no nested groups)
    """
    grps = set([col.split("/")[0] for col in columns])
    grp_columns_dict = {
        grp: [col.split("/")[1] for col in columns if col.startswith(grp + "/")]
        for grp in grps
    }
    return grp_columns_dict




class TestGranule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 42
        random.seed(cls.seed)
        # This is kinda hacky, but okay
        if LOCALFILES_DIR.exists():
            file_pairs = _get_files_local()
        else:
            file_pairs = _get_files_umd()

        # Get verbosity level (default to 1 if not available)
        cls.verbosity = getattr(sys.modules[__name__], "_verbosity", 2)

        # Pick a random subset of file pairs for testing
        # cls.file_pairs = random.sample(file_pairs, k=5)

        cls.file_pairs = file_pairs

    def test_all_shots_present(self):
        for v2file, v3file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing {v2file.name} and {v3file.name}")
            columns = ["shot_number"]
            v2_granule = Granule(v2file, columns)
            v3_granule = Granule(v3file, columns)

            # Check if all shots in V3 are present in V2
            v2_shots = set(v2_granule.data["shot_number"])
            v3_shots = set(v3_granule.data["shot_number"])

            self.assertTrue(
                v3_shots.issubset(v2_shots),
                f"Not all V3 shots are present in V2 for {v2file.name} and {v3file.name}",
            )
            self.assertTrue(
                v2_shots.issubset(v3_shots),
                f"Not all V2 shots are present in V3 for {v2file.name} and {v3file.name}",
            )

    def test_biomass_pi(self):
        for _, v3file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing biomass prediction interval for {v3file.name}")
            columns = [
                "shot_number",
                "agbd",
                "agbd_se",
                "agbd_pi_upper",
                "agbd_pi_lower",
            ]
            v3_granule = Granule(v3file, columns=columns)

            # Check agbd_pi_lower is less than agbd, less than agbd_pi_upper
            v3_agbd = v3_granule.data["agbd"]
            v3_agbd_pi_upper = v3_granule.data["agbd_pi_upper"]
            v3_agbd_pi_lower = v3_granule.data["agbd_pi_lower"]
            self.assertTrue(
                np.all(v3_agbd_pi_lower <= v3_agbd),
                f"agbd_pi_lower is not less than agbd in V3 for {v3file.name}",
            )
            self.assertTrue(
                np.all(v3_agbd <= v3_agbd_pi_upper),
                f"agbd is not less than agbd_pi_upper in V3 for {v3file.name}",
            )

    def test_agbd_quality_flags(self):
        print("\nWARNING: Currently omitting predict_stratum = None")
        for _, file in self.file_pairs:
            qflag = (
                "l4_quality_flag"
                if "V002" in file.name
                else "l4a_quality_flag_rel3"
            )
            columns = [
                "shot_number",
                "agbd",
                "predict_stratum",
                qflag,
            ]
            granule = Granule(file, columns=columns)

            # Check if agbd is not -9999 where quality flags are 1
            valid_mask = granule.data[qflag] == 1
            if "V003" in file.name:
                valid_mask = np.logical_and(
                    valid_mask, granule.data["predict_stratum"] != b"None"
                )
            self.assertTrue(
                np.all(granule.data["agbd"][valid_mask] != -9999),
                f"AGBD is -9999 where quality flag is 1 in V3 for {file.name}",
            )

    def test_agbd_mean_diff(self):
        # I don't think this test is very meaningful.
        big_diffs = []
        for v2file, v3file in self.file_pairs:
            columns = ["shot_number", "agbd"]
            v2_granule = Granule(
                v2file, columns=columns.append("l4_quality_flag")
            )
            v3_granule = Granule(
                v3file, columns=columns.append("l4a_quality_flag_rel3")
            )

            # print mean difference in AGBD between v2 and v3
            v2_agbd = v2_granule.data["agbd"]
            v3_agbd = v3_granule.data["agbd"]
            valid = np.logical_and(v3_agbd != -9999, v2_agbd != -9999)
            mean_diff = np.mean(v3_agbd[valid] - v2_agbd[valid])

            if np.abs(mean_diff) > 5:
                print(
                    f"Mean(V3_agbd - V2_agbd) for {v3file.name} and {v2file.name} is {mean_diff:.2f}"
                )
                big_diffs.append((v2file.name, v3file.name, mean_diff))

        df = pd.DataFrame(
            big_diffs,
            columns=["v2file", "v3file", "mean_diff"],
        )
        # sort by mean diff
        df = df.sort_values(by="mean_diff", ascending=False).to_csv(
            "big_diffs.csv", index=False
        )

        big_diffs = sorted(big_diffs, key=lambda x: np.abs(x[2]), reverse=True)
        for diff in big_diffs[:5]:
            print(diff)

    def test_same_predict_strata(self):
        omit = ["O19336_04", "O19337_04"]
        print(f"\nWARNING: Currently omitting orbits {omit}")
        max_n = 0
        max_fpair = None
        for v2file, v3file in self.file_pairs:
            if any([s in v2file.name for s in omit]):
                continue
            columns = ["shot_number", "predict_stratum"]
            v2g = Granule(v2file, columns)
            v3g = Granule(v3file, columns)
            invalid = np.logical_and(
                v2g.data["predict_stratum"] == b"",
                v3g.data["predict_stratum"] == b"None",
            )
            # print differences in predict_stratum
            v2ps = v2g.data["predict_stratum"][~invalid]
            v3ps = v3g.data["predict_stratum"][~invalid]
            if not np.array_equal(v2ps, v3ps):
                diffs = v2ps != v3ps
                v2p = v2ps[diffs]
                v3p = v3ps[diffs]
                n_diff = len(v2p)
                if n_diff > max_n:
                    max_n = n_diff
                    max_fpair = (v2file.name, v3file.name)
                n_tot = len(v2ps)
                if self.verbosity >= 2:
                    print(f"Differences: {n_diff / n_tot}")
                self.assertLess(n_diff / n_tot, 0.004)

                if self.verbosity >= 2:
                    pairs = set(zip(v2p, v3p))
                    print(pairs)
        if self.verbosity >= 2:
            print(
                (
                    "Max differences in predict_stratum:"
                    f"{max_n} between {max_fpair[0]} and {max_fpair[1]}"
                )
            )

    def test_quality_flag_logic(self):
        qf_columns = [
            "land_cover_data/worldcover_class",
            "land_cover_data/urban_proportion",
            "l2a_quality_flag_rel3",
            "predict_stratum",
            "land_cover_data/leaf_off_flag",
            "l4a_quality_flag_rel3",
        ]

        def get_98_only_models(ancillary):
            idx = ancillary["model_data"]["rh_index"].apply(
                lambda arr: arr[0] == 98 and sum(arr) == 98
            )
            return set(ancillary["model_data"]["predict_stratum"][idx])


        def myqf_l4a(granule: Granule):
            only_98_models = get_98_only_models(granule.ancillary)
            is_98_only = granule.data["predict_stratum"].isin(only_98_models)

            qf = (
                (granule.data["l2a_quality_flag_rel3"] == 1)
                & (granule.data["land_cover_data/urban_proportion"] < 50)
                & (granule.data["land_cover_data/worldcover_class"] != 0)
                & (granule.data["land_cover_data/worldcover_class"] != 80)
                & (granule.data["predict_stratum"] != b"None")
                & (
                    (granule.data["land_cover_data/leaf_off_flag"] == 0)
                    | (granule.data["land_cover_data/leaf_off_flag"] == 255)
                    | (is_98_only)
                )
            )
            return qf

        def myqf_l2a(granule: Granule):
            return granule.data["rh"].apply(lambda arr: arr[100] < 150)
        
        if self.verbosity >= 2:
            print("As a reminder, the RH98-only models are:")
            g = Granule(self.file_pairs[0][1], columns=["agbd"])
            print(get_98_only_models(g.ancillary))

        ### TEST STARTS HERE ###
        print("WARNING: Currently accepting leaf_off_flag = 255 as valid")
        for _, v3file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing quality flag logic for {v3file.name}")
            l2a_file = _get_l2a_file_umd(v3file)

            g = Granule(v3file, columns=qf_columns)
            g_l2a = Granule(l2a_file, columns=["rh"])

            myqfl4a_set = myqf_l4a(g)
            myqfl2a_set = myqf_l2a(g_l2a)
            myqf = myqfl4a_set & myqfl2a_set
            qf_v3 = g.data["l4a_quality_flag_rel3"] == 1

            diff = myqf != qf_v3
            self.assertFalse(
                diff.any(),
                f"Quality flag logic mismatch in {v3file.name}."
            )
            # if self.verbosity >= 2:
            #     print(
            #         "MyQF (L4A) vs V3 QF:\n",
            #         g.data[diff][["shot_number", "predict_stratum"]],
            #     )
            #     g.data[diff].to_csv(
            #         "quality_flag_mismatch.csv", index=False
            #     )

    def test_all_metadata_fields(self):
        md = DATA_DICTIONARY_DIR / "product_dictionary_metadata.txt"
        columns = _get_columns(md)
        for _, file in self.file_pairs:
            f = h5py.File(file, "r")
            for col in columns:
                if col not in f.keys():
                    self.fail(f"Column {col} not found in {file.name}")

    def test_all_ancillary_fields(self):
        ancillary = DATA_DICTIONARY_DIR / "product_dictionary_ancillary.txt"
        columns = _get_columns(ancillary)
        grp_columns_dict = _get_grps_columns_dict(columns)
        grps = grp_columns_dict.keys()
        columns = [col.split("/") for col in columns]
        for _, file in self.file_pairs:
            g = Granule(file, columns=["agbd"])
            for grp, name in columns:
                if grp not in g.ancillary.keys():
                    self.fail(f"Group {grp} not found in {file.name}")
                if name not in g.ancillary[grp].columns:
                    self.fail(
                        f"Column {name} not found in group {grp} of {file.name}"
                    )
            self.assertEqual(
                len(set(g.ancillary["model_data"]["predict_stratum"])), 35
            )
            self.assertSetEqual(
                grps,
                set(g.ancillary.keys()),
                f"Ancillary groups in {file.name} do not match expected groups",
            )

            for grp in grps:
                self.assertSetEqual(
                    set(g.ancillary[grp].columns),
                    set(grp_columns_dict[grp]),
                    f"Columns in group {grp} of {file.name} do not match expected columns",
                )
                for col in g.ancillary[grp].columns:
                    self.assertFalse(
                        g.ancillary[grp][col].isnull().any(),
                        f"Column {col} in group {grp} of {file.name} contains null values",
                    )

    def test_all_beam_attributes(self):
        beam_attrs = DATA_DICTIONARY_DIR / "product_dictionary_beam_attrs.txt"
        columns = _get_columns(beam_attrs)
        add = [
            "elev_outlier_zscore_min",
            "elev_outlier_tile_buffer",
            "elev_outlier_zq",
            "elev_outlier_zscore_t",
            "elev_outlier_ncells",
            "elev_outlier_tile_res",
            "elev_outlier_zdiff",
            "elev_outlier_tile_size",
        ]
        print("\nWARNING: Currently adding the following columns:")
        print(add)
        columns.extend(add)
        for _, file in self.file_pairs:
            f = h5py.File(file, "r")
            for k in f.keys():
                if k.startswith("BEAM"):
                    attrs = f[k]["agbd_prediction"].attrs.keys()
                    self.assertSetEqual(
                        set(attrs),
                        set(columns),
                        f"Attributes for {k} in {file.name} do not match expected columns",
                    )

    def test_all_beam_fields(self):
        omit = ["geolocation/degrade_flag", "land_cover_data/pft_class"]
        print("\nWARNING: Currently omitting the following fields:")
        print(omit)
        beam_data = DATA_DICTIONARY_DIR / "product_dictionary_beam_data.txt"
        columns = _get_columns(beam_data)
        columns = [c for c in columns if c not in omit]
        all_granules_all_zero_cols = {c: 0 for c in columns}
        for _, file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing beam fields for {file.name}")
            g = Granule(file, columns=columns)
            dat = g.data
            for col in dat.columns:
                self.assertFalse(
                    dat[col].isnull().all(),
                    f"Column {col} of {file.name} is all null.",
                )
                if is_numeric_dtype(dat[col]):
                    if (dat[col] == 0).all():
                        all_granules_all_zero_cols[col] += 1
        for col, count in all_granules_all_zero_cols.items():
            if count == len(self.file_pairs):
                print(f"\nWARNING: column {col} is all zero in all granules.")

    def test_no_extra_beam_fields(self):
        print("\nWARNING: Currently ignoring degrade_flag")
        ignore = ["degrade_flag", "geolocation/degrade_flag"]
        beam_data = DATA_DICTIONARY_DIR / "product_dictionary_beam_data.txt"
        columns = _get_columns(beam_data)
        columns = [c for c in columns if c not in ignore]
        grp_columns_dict = _get_grps_columns_dict(columns)
        grps = grp_columns_dict.keys()
        columns = [col.split("/") for col in columns]

        for _, file in self.file_pairs:
            f = h5py.File(file, "r")
            for k in f.keys():
                if k.startswith("BEAM"):
                    beam_grps = set(f[k].keys())
                    for ig in ignore:
                        beam_grps.discard(ig)
                    self.assertSetEqual(
                        beam_grps,  # actual
                        set(grps),  # documentation
                        f"Groups in {k} of {file.name} do not match expected groups",
                    )
                    for grp in beam_grps:
                        if not grp_columns_dict[grp]:
                            # This is a top-level group with no subcolumns
                            continue
                        self.assertSetEqual(
                            set(f[k][grp].keys()),  # actual
                            set(grp_columns_dict[grp]),  # documentation
                            f"Columns in group {grp} of {k} in {file.name} do not match expected columns",
                        )
    


if __name__ == "__main__":
    unittest.main()
