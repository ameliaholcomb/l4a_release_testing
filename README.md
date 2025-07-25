## Release tests for L4A V3

`granules_test.py` contains all tests.

To run all tests (can take a long time), use
```bash
$ cd /path/to/repository
$ python -m unittest test_granules
```

To run any individual test, use
```bash
$ cd /path/to/repository
$ python -m unittest test_granules.TestGranule.[name_of_test]
```

Set verbosity to level 1, 2, or 3 by adding `-v` flags, e.g. `-vvv` for maximum debug information printed.

These tests should run from the UMD cluster, however for speed if you have a local SSD I highly recommend copying the data to the SSD and running the tests locally. 
In that case, set `LOCALFILES_DIR` at the top of `test_granules.py`

Some tests are more qualitative. 
`examine_granules.ipynb` contains graphs of the data and may be a useful starting point.
