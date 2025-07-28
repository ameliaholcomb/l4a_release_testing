import pathlib
from typing import List
import h5py
import pandas as pd


class Granule(h5py.File):
    """
    A class to represent a GEDI granule.
    """

    def __init__(
        self,
        file_path: pathlib.Path,
        columns: List[str],
        quality_filter: bool = False,
    ):
        super().__init__(file_path, "r")
        self.beams = list([k for k in self.keys() if k.startswith("BEAM")])
        if not columns:
            # TODO: Ideally, get all columns if none are specified
            raise NotImplementedError
        if "shot_number" not in columns:
            columns.append("shot_number")
        if quality_filter:
            if file_path.name.endswith("V002.h5"):
                if "l4_quality_flag" not in columns:
                    columns.append("l4_quality_flag")
            elif file_path.name.endswith("V003.h5"):
                if "l4a_quality_flag_rel3" not in columns:
                    columns.append("l4a_quality_flag_rel3")
                # Currently working around a bug in the V003 files where
                # the predict_stratum is sometimes not present but quality flag
                # is True.
                if "predict_stratum" not in columns:
                    columns.append("predict_stratum")

        self.columns = columns
        self.data = self.iter_beams()
        self.ancillary = self.iter_ancillary()
        if quality_filter:
            if "l4_quality_flag" in self.data.columns:
                self.data = self.data[self.data["l4_quality_flag"] == 1]
            elif "l4a_quality_flag_rel3" in self.data.columns:
                # Currently working around a bug in the V003 files where
                # the predict_stratum is sometimes not present but quality flag
                # is True.
                self.data = self.data[
                    (self.data["l4a_quality_flag_rel3"] == 1)
                    & (self.data["predict_stratum"] != b'None')
                ]

    def iter_ancillary(self):
        """
        Iterate over the ancillary data in the granule and extract data.
        """
        ancillary_data = {}
        if "ANCILLARY" not in self:
            return ancillary_data
        for key in self["ANCILLARY"]:
            d = self["ANCILLARY"][key]
            if isinstance(d, h5py.Dataset):
                columns = self["ANCILLARY"][key].dtype.fields.keys()
                dic = {col: list(d[col][:]) for col in columns}
                df = pd.DataFrame(dic)
            ancillary_data[key] = df
        return ancillary_data

    def iter_beams(self):
        """
        Iterate over the beams in the granule and extract data.
        """
        data = []
        for beam in self.beams:
            beam_data = {}
            for col in self.columns:
                if col in self[beam]:
                    d = self[beam][col][:]
                    if len(d.shape) == 1:
                        beam_data[col] = d
                    else:
                        beam_data[col] = list(d)
                else:
                    raise ValueError(f"Requested field {col} not found.")
            beam_data["beam"] = beam
            data.append(beam_data)
        data = pd.concat([pd.DataFrame(d) for d in data], ignore_index=True)
        data.set_index("shot_number", inplace=True, drop=False)

        return data
