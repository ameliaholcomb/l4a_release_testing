import h5py
from collections import defaultdict
import numpy as np
import pathlib
import random
import sys
import unittest
from typing import Dict, List

from tools.granule import Granule
from tools.granule_name import parse_granule_filename
from tools.null_checker import check_null, check_zero

LOCALFILES_DIR = pathlib.Path(
    "/Users/amelia/Repos/testdata/gedi_l4a_test_granules"
)
V3FILES_DIR = pathlib.Path(
    "/gpfs/data1/vclgp/armstonj/gedi_l4a_processing/r003"
)
V2FILES_DIR = pathlib.Path("/gpfs/data1/vclgp/data/iss_gedi/soc/2022/132")


TESTDIR = pathlib.Path(__file__).parent
DATA_DICTIONARY_DIR = TESTDIR / "data_dictionaries"


def _get_files_umd():
    v3files = list(V3FILES_DIR.iterdir())
    v2files = [_get_v2_file_umd(f) for f in v3files]
    return list(zip(v2files, v3files))


def _get_v2_file_umd(v3file: pathlib.Path):
    md = parse_granule_filename(v3file.name)
    v2f = list(
        V2FILES_DIR.glob(f"GEDI04_A*{md.orbit}_{md.sub_orbit_granule}*_V002.h5")
    )
    if len(v2f) == 2:
        if v2f[0] > v2f[1]:
            return v2f[0]
        else:
            return v2f[1]
    elif len(v2f) == 1:
        return v2f[0]
    else:
        print("Error: unexpected number of files found\n", v2f)


def _get_l2a_file_umd(v3file: pathlib.Path):
    md = parse_granule_filename(v3file.name)
    l2afiles = list(
        V2FILES_DIR.glob(f"GEDI02_A*{md.orbit}_{md.sub_orbit_granule}*_V003.h5")
    )
    l2afiles = sorted(l2afiles, reverse=True)
    return l2afiles[0]


def _get_files_local():
    v3files = sorted(list(LOCALFILES_DIR.glob("*V003.h5")))
    v2files = sorted(list(LOCALFILES_DIR.glob("*V002.h5")))
    return list(zip(v2files, v3files))


def _get_columns(file_path: pathlib.Path):
    columns = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith("_aN"):
                columns.extend(
                    [line.replace("_aN", f"_a{i}") for i in [1, 2, 5, 10]]
                )
            else:
                columns.append(line)
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
        if sys.argv and len(sys.argv) > 2 and "-v" in sys.argv[2]:
            cls.verbosity = sys.argv[2].count("v") - 1
        else:
            cls.verbosity = 1
        print(f"Verbosity level: {cls.verbosity}")

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
                f"Some V3 shots are not in V2 for {v2file.name}, {v3file.name}",
            )
            self.assertTrue(
                v2_shots.issubset(v3_shots),
                f"Some V2 shots are not in V3 for {v2file.name}, {v3file.name}",
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
        for _, file in self.file_pairs:
            columns = [
                "shot_number",
                "agbd",
                "l4a_quality_flag_rel3",
            ]
            granule = Granule(file, columns=columns)
            # Check if agbd is not -9999 where quality flags are 1
            valid_mask = granule.data["l4a_quality_flag_rel3"] == 1
            self.assertTrue(
                np.all(granule.data["agbd"][valid_mask] != -9999),
                f"AGBD is -9999 where quality flag is 1 in V3 for {file.name}",
            )

    def test_same_predict_strata(self):
        max_n = 0
        max_fpair = None
        THRESHOLD = 0.004  # 0.4% of footprints
        for v2file, v3file in self.file_pairs:
            columns = ["shot_number", "predict_stratum"]
            v2g = Granule(v2file, columns)
            v3g = Granule(v3file, columns)
            # print differences in predict_stratum
            v2ps = v2g.data["predict_stratum"]
            v3ps = v3g.data["predict_stratum"]
            if not np.array_equal(v2ps, v3ps):
                diffs = v2ps != v3ps
                v2p = v2ps[diffs]
                n_diff = len(v2p)
                if n_diff > max_n:
                    max_n = n_diff
                    max_fpair = (v2file.name, v3file.name)
                n_tot = len(v2ps)
                if self.verbosity >= 2 and (n_diff / n_tot) > THRESHOLD:
                    print(f"Differences: {n_diff / n_tot}")
                    v3p = v3ps[diffs]
                    pairs = list(zip(v2p, v3p))
                    for u in set(pairs):
                        print(f"{u[0]} -> {u[1]}: {pairs.count(u)} times")
                self.assertLess(
                    n_diff / n_tot,
                    THRESHOLD,
                    f"{n_diff / n_tot * 100:.2f}% of footprints differ in "
                    f"predict_stratum between {v2file.name} and {v3file.name}",
                )
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
                & (granule.data["predict_stratum"] != b"")
                & (
                    (granule.data["land_cover_data/leaf_off_flag"] == 0)
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
        for _, v3file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing quality flag logic for {v3file.name}")
            l2a_file = _get_l2a_file_umd(v3file)

            g = Granule(v3file, columns=qf_columns)
            g_l2a = Granule(l2a_file, columns=["rh"])

            myqf = myqf_l4a(g) & myqf_l2a(g_l2a)
            qf_v3 = g.data["l4a_quality_flag_rel3"] == 1

            diff = myqf != qf_v3
            if self.verbosity >= 2 and diff.any():
                print("Saving quality flag issues to quality_flag_issues.csv")
                g.data[diff].to_csv("quality_flag_issues.csv", index=False)
            self.assertFalse(
                diff.any(), f"Quality flag logic mismatch in {v3file.name}."
            )

    def test_all_metadata_fields(self):
        md = DATA_DICTIONARY_DIR / "product_dictionary_metadata.txt"
        columns = _get_columns(md)
        grps_columns_dict = _get_grps_columns_dict(columns)
        grps = grps_columns_dict.keys()
        for _, file in self.file_pairs:
            print(file)
            f = h5py.File(file, "r")
            self.assertSetEqual(set(f.attrs.keys()), set(["short_name"]))
            metadata = f["METADATA"]
            for grp in grps:
                if grp not in metadata.keys():
                    self.fail(f"Group {grp} not found in {file.name}")
                for att in grps_columns_dict[grp]:
                    if att not in metadata[grp].attrs.keys():
                        self.fail(
                            f"Attribute {att} not found in group {grp} of {file.name}"
                        )
                    self.assertIsNotNone(metadata[grp].attrs[att])
            self.assertSetEqual(
                set(grps),
                set(metadata.keys()),
                f"Metadata groups in {file.name} do not match expected groups",
            )

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
                set(grps),
                set(g.ancillary.keys()),
                f"Ancillary groups in {file.name} do not match expected groups",
            )

            for grp in grps:
                self.assertSetEqual(
                    set(g.ancillary[grp].columns),
                    set(grp_columns_dict[grp]),
                    f"Columns in ANCILLARY/{grp} in {file.name} don't match expected columns",
                )
                for col in g.ancillary[grp].columns:
                    self.assertFalse(
                        check_null(g.ancillary[grp][col]),
                        f"ANCILLARY/{grp}/{col} of {file.name} has null values",
                    )

    def test_all_beam_attributes(self):
        beam_attrs = DATA_DICTIONARY_DIR / "product_dictionary_beam_attrs.txt"
        columns = _get_columns(beam_attrs)
        for _, file in self.file_pairs:
            f = h5py.File(file, "r")
            for k in f.keys():
                if k.startswith("BEAM"):
                    attrs = f[k]["agbd_prediction"].attrs.keys()
                    self.assertSetEqual(
                        set(attrs),
                        set(columns),
                        (
                            f"Attributes for {k}/agbd_prediction in {file.name}"
                            " do not match expected columns"
                        ),
                    )

    def test_no_extra_beam_fields(self):
        beam_data = DATA_DICTIONARY_DIR / "product_dictionary_beam_data.txt"
        columns = _get_columns(beam_data)
        grp_columns_dict = _get_grps_columns_dict(columns)
        grps = grp_columns_dict.keys()
        columns = [col.split("/") for col in columns]

        for _, file in self.file_pairs:
            f = h5py.File(file, "r")
            for k in f.keys():
                if k.startswith("BEAM"):
                    beam_grps = set(f[k].keys())
                    self.assertSetEqual(
                        beam_grps,  # actual
                        set(grps),  # documentation
                        f"Groups in {k} of {file.name} do not match expected",
                    )
                    for grp in beam_grps:
                        if not grp_columns_dict[grp]:
                            # This is a top-level group with no subcolumns
                            continue
                        self.assertSetEqual(
                            set(f[k][grp].keys()),  # actual
                            set(grp_columns_dict[grp]),  # documentation
                            (
                                f"Columns in {k}/{grp} of {file.name}"
                                " do not match expected"
                            ),
                        )

    def test_all_beam_fields(self):
        beam_data = DATA_DICTIONARY_DIR / "product_dictionary_beam_data.txt"
        columns = _get_columns(beam_data)
        all_null_cols = defaultdict(int)
        all_zero_cols = defaultdict(int)
        for _, file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing non-null data for {file.name}")
            g = Granule(file, columns=columns)
            for col in g.data.columns:
                all_null_cols[col] += check_null(g.data[col])
                all_zero_cols[col] += check_zero(g.data[col])
        k_files = len(self.file_pairs)
        bad = [col for col, n in all_zero_cols.items() if n == k_files]
        if len(bad) > 0:
            print(f"WARNING: Columns are zero across all granules:\n{bad}")
        bad = [col for col, n in all_null_cols.items() if n == k_files]
        self.assertListEqual(bad, [], "\nColumns are null across all granules")

    def test_binary_flags_binary_values(self):
        binary_flags = [
            "l4a_quality_flag_rel3",
            "l2a_quality_flag_rel3",
            "degrade_include_flag",
            "surface_flag",
            "geolocation/stale_return_flag",
        ]
        binary_nullable_flags = [
            "land_cover_data/leaf_off_flag",
        ]
        zeroonetwo_nullable_flags = [
            "predictor_limit_flag",
            "response_limit_flag",
        ]
        for _, file in self.file_pairs:
            if self.verbosity >= 2:
                print(f"Testing binary flags for {file.name}")
            g = Granule(
                file,
                columns=binary_flags
                + binary_nullable_flags
                + zeroonetwo_nullable_flags,
            )
            for col in binary_flags:
                self.assertTrue(
                    np.all(np.isin(g.data[col], [0, 1])),
                    f"{col} in {file.name} contains {g.data[col].unique()}.",
                )
            for col in binary_nullable_flags:
                self.assertTrue(
                    np.all(np.isin(g.data[col], [0, 1, 255])),
                    f"{col} in {file.name} contains {g.data[col].unique()}.",
                )
            for col in zeroonetwo_nullable_flags:
                self.assertTrue(
                    np.all(np.isin(g.data[col], [0, 1, 2, 255])),
                    f"{col} in {file.name} contains {g.data[col].unique()}.",
                )


if __name__ == "__main__":
    unittest.main()
