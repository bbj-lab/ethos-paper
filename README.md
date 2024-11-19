# Partitioned Inference

This branch adds the option to split long-running inference jobs into multiple
parts. It introduces two new options to the `ethos infer` command:

- `--n_parts` this int is the number of parts you want to split the test set
  into
- `--ith_part` this int designates the currently running part; it should be in
  `range(n_parts)`

For each repetition, you'll need to fix `n_parts` and then run an inference job
for `ith_part` over `range(n_parts)`.

We could then run an [array job](https://slurm.schedmd.com/job_array.html) that
would look something like this:

```
...
#SBATCH --array=0-4
...

ethos infer \
    ...
    --ith_part ${SLURM_ARRAY_TASK_ID} \
    --n_parts 5 \
    ...
```
