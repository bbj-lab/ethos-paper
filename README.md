# Partitioned Inference

This branch adds the option to split long-running inference jobs into multiple
parts. It introduces two new options to the `ethos infer` command:

- `--n_parts` this int is the number of parts you want to split the test set
  into
- `--ith_part` this int designates the currently running part; it should be in
  `range(n_parts)`

For each repetition, you'll need to fix `n_parts` and then run an inference job
for `ith_part` over `range(n_parts)`.

> Note: there is an unlikely but potential race condition on the results file
> if multiple `ith_part` parts finish at exactly the same time. This condition
> can be avoided by setting up an
> [array job](https://slurm.schedmd.com/job_array.html) that uses `%1` to make
> sure only one part runs at once. For example, with 5 parts, we would have
> something like this:

```
...
#SBATCH --array=0-4%1
...

ethos infer \
    ...
    --ith_part ${SLURM_ARRAY_TASK_ID} \
    --n_parts 5 \
    ...
```

In the standard version of ethos, if you had all the results files at the end
of a run, then the inference job had been successful. _That is no longer the
case with this code: if e.g. one of the jobs fails on a 5-part split, then you
will have all of the results files, but they will be ~4/5ths the size of what
they ought to be._ Please be careful.
