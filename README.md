## Release tests for L4A V3

`test_granules.py` contains all tests.

These tests can be run from the UMD cluster, in which case the V3 files under test should be located in `V3FILES_DIR` (set at the top of the file).
The L2A_v3 and L4A_v2 files will be automatically loaded based on file name matching from the orbit directory `V2FILES_DIR`

However, for speed if you have a local SSD I highly recommend copying all the data to the SSD and running the tests locally. 
In that case, set `LOCALFILES_DIR`.

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
Except when debugging an individual test failure, it is recommended to set verbosity to level 1 only (or omit the flag) in order to make sure that important warnings are not missed.

On branch `known-issues`, all known issues are printed as warnings instead of test failures.
This may be useful for debugging and writing new tests.

Some tests are more qualitative. 
`examine_granules.ipynb` contains graphs of the data and may be a useful starting point.
