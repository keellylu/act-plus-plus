import h5py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=str, help="Path to the demo .hdf5 file")
    parser.add_argument("--key", type=str, default=None,
                        help="Optional dataset key for actions (if not provided, script will search).")
    args = parser.parse_args()

    path = args.hdf5_path

    with h5py.File(path, "r") as f:
        print("Available top-level keys:")
        for k in f.keys():
            print("  -", k)

        # If the user didn't specify a dataset key, try common action dataset names.
        if args.key is None:
            common_keys = ["actions", "action", "policy_action", "expert_action"]
            found_key = None
            for ck in common_keys:
                if ck in f:
                    found_key = ck
                    break
            if found_key is None:
                raise ValueError(
                    "Could not find an actions dataset. Please supply --key <dataset_name>."
                )
            key = found_key
        else:
            key = args.key
            if key not in f:
                raise ValueError(f"Dataset key '{key}' not found in file.")

        actions = f[key]
        print(f"\nDataset '{key}' found.")
        print("Shape:", actions.shape)
        print("Length:", actions.shape[0])

if __name__ == "__main__":
    main()
