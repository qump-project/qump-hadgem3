import os
import pathlib
import re
import argparse
import logging
from collections import defaultdict
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

class PPEDataFrame(pd.DataFrame):
    """
    Custom DataFrame class for handling PPE data.
    Inherits from pandas.DataFrame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return PPEDataFrame

    @staticmethod
    def _convert_raw_to_ppe(raw_df):
        """
        Convert a raw DataFrame to a PPEDataFrame with appropriate metadata and indexing.
        """
        metadata_df = raw_df[raw_df['ROW TYPE'] == "Parameter"]
        ppe_raw_df = raw_df[raw_df['ROW TYPE'] == "Experiment"]

        metadata_dropcols = ['ROW TYPE', 'Realisation', 'Initialisation', 'Parameters']
        metadata_df = metadata_df.drop(columns=metadata_dropcols)
        metadata_df.set_index("ATTRIBUTE_NAME", inplace=True)

        ppe_raw_df = ppe_raw_df.convert_dtypes()
        ppe_raw_df['RIPCODE'] = ppe_raw_df.apply(lambda row: "r{0:03d}i{1:01d}p{2:05d}".format(row['Realisation'], row['Initialisation'], row['Parameters']), axis=1)
        ppe_raw_df.set_index("RIPCODE", inplace=True)

        ppe_dropcols = ['ROW TYPE', 'ATTRIBUTE_NAME', 'Realisation', 'Initialisation', 'Parameters']
        ppe_raw_df = ppe_raw_df.drop(columns=ppe_dropcols)
        ppe_raw_df = ppe_raw_df.apply(pd.to_numeric)

        assert set(metadata_df.columns).symmetric_difference(ppe_raw_df.columns) == set()

        ppe_df = ppe_raw_df
        for param_name in ppe_df.columns:
            ppe_df[param_name].attrs = metadata_df[param_name].to_dict()

        return ppe_df

    @classmethod
    def read_csv(cls, *args, **kwargs):
        """
        Read a CSV file into a PPEDataFrame.
        """
        return cls(pd.read_csv(*args, **kwargs))

    @classmethod
    def read_PPE_csv(cls, *args, **kwargs):
        """
        Read a CSV file and convert it to a PPEDataFrame.
        """
        raw_df = cls(pd.read_csv(*args, **kwargs))
        return cls._convert_raw_to_ppe(raw_df)

    def get_rip_r(self):
        """
        Extract the 'Realisation' part from the RIPCODE index.
        """
        return [int(rip[1:4]) for rip in self.index]

    def get_rip_i(self):
        """
        Extract the 'Initialisation' part from the RIPCODE index.
        """
        return [int(rip[5:6]) for rip in self.index]

    def get_rip_p(self):
        """
        Extract the 'Parameters' part from the RIPCODE index.
        """
        return [int(rip[7:]) for rip in self.index]

    def get_rzn(self):
        """
        Generate a unique realization number from the RIPCODE index.
        """
        return [(10**5)*int(rip[1:4]) + (10**4)*int(rip[5:6]) + int(rip[7:]) for rip in self.index]


def remove_values_from_list(list_of_values, value_to_remove):
    return [value for value in list_of_values if value != value_to_remove]


def check_and_make_dir(dirname):
    """
    Function that makes a directory and doesn't throw an error if it already
    exists
    """
    dir_path = pathlib.Path(dirname)
    dir_path.mkdir(parents=True, exist_ok=True)


def generate_namelist_param_dict(ppe_df, rip):
    """
    Generate a dictionary with keys as namelist names and values as lists of parameters under that namelist.
    """
    namelist_param_dict = defaultdict(dict)
    ppe_df_row = ppe_df.loc[rip]

    for param_name in ppe_df_row.index:
        param_metadata = ppe_df[param_name].attrs
        param_namelist = param_metadata["namelist"]
        param_value = ppe_df_row[param_name]
        
        if param_metadata["printInNamelist"] == "TRUE":
            param_value_str = "{0:#.5g}".format(param_value)
            namelist_param_dict[param_namelist][param_name] = param_value_str

    return namelist_param_dict


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to CSV file that defines the parameter perturbations")
    return parser.parse_args()


def load_ppe_dataframe(csv_file):
    """
    Load the PPE DataFrame from the input CSV file.
    """
    return PPEDataFrame.read_PPE_csv(csv_file)


def get_rip_list():
    """
    Get list of RIPCODES to produce opt files for from environment variable.
    """
    rip_list_raw = os.environ.get("ENS_RIPCODES", "")
    rip_list = re.sub(r"\(|\)|'|\s+", "", rip_list_raw).split(",")
    return remove_values_from_list(rip_list, "")


def create_opt_files(ppe_df, rip_list, opt_file_dir, params_to_ignore):
    """
    Generate opt files for each RIPCODE in the list.
    """
    for rip in rip_list:
        logging.info(f"Processing RIPCODE: {rip}")
        namelist_param_dict = generate_namelist_param_dict(ppe_df, rip)
        opt_file_fname = f"rose-app-{rip}.conf"
        opt_file_fpath = os.path.join(opt_file_dir, opt_file_fname)

        with open(opt_file_fpath, "w") as opt_file_obj:
            for namelist_name, param_dict in namelist_param_dict.items():
                namelist_str = f"[namelist:{namelist_name}]\n"
                opt_file_obj.write(namelist_str)

                param_list = sorted(param_dict.keys())
                for param_name in param_list:
                    if param_name in params_to_ignore:
                        continue
                    param_value_str = param_dict[param_name]
                    param_str = f"{param_name}={param_value_str}\n"
                    opt_file_obj.write(param_str)

                opt_file_obj.write("\n")

        logging.info(f"Opt file written for {opt_file_fpath}")


def main():
    """
    Main function to orchestrate the generation of opt files.
    """
    args = parse_arguments()
    ppe_df = load_ppe_dataframe(args.csv_file)
    params_to_ignore = []

    rip_list = get_rip_list()
    logging.info(f"RIPCODES to process: {rip_list}")

    opt_file_dir = os.path.join(os.environ.get("LINUX_OPT_DIR", ""))
    check_and_make_dir(opt_file_dir)

    create_opt_files(ppe_df, rip_list, opt_file_dir, params_to_ignore)


if __name__ == "__main__":
    main()