#!/bin/bash

logfile_name="terminal_test_load_model_$(date "+%Y%m%d_%H%M%S").log"

python3 test_load_models.py |& tee "$logfile_name"
# python3 test_load_models.py --model-dir-from-file failed_models_20240104_210227.log |& tee "$logfile_name"

# python3 test_load_models.py --check-stderr |& tee "$logfile_name"
# python3 test_load_models.py --check-stderr --model-dir-from-file failed_models_20240104_231350.log |& tee "$logfile_name"
