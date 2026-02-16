from clean_data import clean_data
from filter_papers_gemini import filter_papers_gemini
import argparse

TEST_FOLDER = "extraction-test"
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder_name", default=TEST_FOLDER, help="Extraction folder name under data/raw/ to process (e.g. extraction-1)")
    args = p.parse_args()
    folder_name = args.folder_name

    clean_data(folder_name)
    filter_papers_gemini(folder_name)


if __name__ == "__main__":
    main()