
# Learning to Encode and Classify Test Executions (ICSE 2020)

This is a complementary repository for the paper submitted to ICSE 2020, "Learning to Encode and Classify Test Executions". It contains the code for each step of the approach explained, and includes the original execution traces of one of the case studies (Ethereum).

Using the scripts it is possible to train a neural network for classification and recreate the results presented for that specific program.

## Files

```
1. instrumentation_pass/        (LLVM instrumentation pass folder)
2. ethereum_traces/             (Folder with the execution traces)
3. preprocess.py                (Performs preprocessing on the original traces)
4. train.py                     (Framework for training the proposed architecture)
5. run.sh                       (Automated script)
6. fsm_ablation_study.pdf       (Ablation study over all 10 FSM protocols)
```
## Usage

The instrumentation pass (``func_call_rec.cpp``) requires ``LLVM 8.0.0`` and ``Clang 8.0.0`` installed.
Inside the ``instrumentation_pass/`` folder, the source code for the external instrumentation library that stores the trace information is also included (``external_lib.cpp``, ``external_lib.h``, ``function_call.h``, ``ir_type.h``) . 
We include the source code and the traces gathered using these tools.

In order to pre-process, encode and use the traces to train the neural network architecture run ``run.sh``.

Both Python scripts require ``Python 3.7`` and ``Pytorch`` installed.
 
