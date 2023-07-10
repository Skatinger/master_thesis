import re
import sys
import json
from evaluation.loader import ResultLoader


def main():
    assert len(sys.argv) > 1, "Please provide a path to the result file as an argument"

    key = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    # change this if only one config was used
    configs = ['original']

    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_rulings_gt()

    if model_name is not None:
        results = loader.load(key, model_name)
    else:
        results = loader.load(key)

    import pdb; pdb.set_trace()

    # find predictions which are actually names and seem to be correct


    # write results to json file
    file_name = f"{key}-results" if model_name is None else f"{key}-{model_name}-results"
    save_path = f"evaluation/results/{file_name}"
    print(f"Writing results to {save_path}")

    # with open(f"{save_path}.json", 'w') as f:
    #     json.dump(json_results, f, indent=4)
    
    # # write results to csv file
    # with open(f"{save_path}.csv", 'w') as f:
    #     f.write('\n'.join(csv_lines))



if __name__ == "__main__":
    main()