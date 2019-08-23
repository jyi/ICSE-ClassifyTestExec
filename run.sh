mkdir ethereum_traces/reduced_traces
mkdir ethereum_traces/reduced_traces/pass
mkdir ethereum_traces/reduced_traces/fail_boolean
mkdir ethereum_traces/reduced_traces/fail_loop
mkdir ethereum_traces/reduced_traces/fail_relational
mkdir ethereum_traces/reduced_traces/fail_swapped_args
mkdir ethereum_traces/reduced_traces/fail_wrong_numbers
mkdir ethereum_traces/encoded_traces
mkdir ethereum_traces/encoded_traces/fail_boolean
mkdir ethereum_traces/encoded_traces/fail_loop
mkdir ethereum_traces/encoded_traces/fail_relational
mkdir ethereum_traces/encoded_traces/fail_swapped_args
mkdir ethereum_traces/encoded_traces/fail_wrong_numbers
python3 preprocess.py && python3 train.py