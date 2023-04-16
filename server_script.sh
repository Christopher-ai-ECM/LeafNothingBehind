#!/usr/bin/bash

# USAGE : sudo ./server_script.sh <path_to_csv> <path_to_inference_results_folder> <path_to_submission_as_zip>

# Check if the zip file exists
if [ ! -f "$3" ]; then
  echo "Zip file not found"
  exit 1
fi

# Unzip data in the test folder
TEST_FOLDER="evaluated_team_project"
rm -rf "$TEST_FOLDER"
mkdir "$TEST_FOLDER"
unzip -q "$3" -d "$TEST_FOLDER"
cd $TEST_FOLDER
ls

# Check if main.py and requirements.txt exist
if [ -f "main.py" ] && [ -f "requirements.txt" ] && [ ! -f "infer.sh" ]; then
  # Install requirements and call main.py
  pip install -r "requirements.txt"
  python "main.py" --mode infer --csv_path "$1" --save_infers_under "$2"
  cd ..
  exit 0
fi

# Check if infer.sh exists
if [ -f "infer.sh" ]; then
  # Call infer.sh
  sudo "./infer.sh" --mode infer --csv_path "$1" --save_infers_under "$2"
  cd ..
  exit 0
fi

# If none of the above conditions are met, print an error message and exit
echo "Invalid test folder contents"
cd ..
exit 1
