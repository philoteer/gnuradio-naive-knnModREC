** Dependency: gr-inspector (https://github.com/gnuradio/gr-inspector), Pandas (Python), Scikit (Python)

License: GPLv3 (Follows GNU Radio's licensing terms)

A naive kNN based automatic modulation classifier (features: BW, PSD_est(max), PSD_est(max) - PSD_est(min)).

- test.py: Validates the classifier by using 80% of the data from feature.csv to train and rest of them to test. Does not need GNU Radio to function.
