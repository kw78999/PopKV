# Measurement provenance

The retained five-task results were measured in the August 2026 experiment and reaggregated for this release. Predictions, scores, and per-sample timing values were preserved. Task selection and the public output-directory name in `experiment_config.json` describe the released subset. `sample_manifest.json` identifies its 950 samples.

`environment.txt` preserves the recorded Python, PyTorch, Transformers, GPU, driver, and baseline Git commit. The historical working-tree status listing was omitted because it refers to obsolete output paths. `pip_freeze.txt` is the recorded package snapshot.

`source_diff.patch` preserves only the historical core implementation changes (`popkv.py`, `popkv_cuda.py`, `fier_triton.py`, `sparse_kernels.py`, and `csrc/`) relative to commit `79ab85de19489b3f83b5ec4fa62e29cec656d1d9`. It is deliberately a core-only patch, not a complete source snapshot: documentation, evaluator changes, and untracked files are not included. The current evaluator is shipped as source in `experiments/evaluate_longbench.py`.

The model revision recorded by the run is `d10aef7999a2b5ba950ab3974312feeedbfe0b77`. A tokenizer revision was not captured. The measurements use Llama-3.1-8B-Instruct with a 32,768-token prompt-truncation setting before chat-template application.

All published means are recomputed from the retained raw records. No model inference was performed during this release cleanup. Use the root README command to run a new evaluation; exact wall times depend on the runtime environment.

FIER labels and implementation identifiers are normalized to the release naming throughout the records and reference patch. Numerical measurements are unchanged. Use the shipped source files rather than applying this normalized reference patch verbatim to the historical tree.
