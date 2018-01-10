** Dependency: gr-inspector (https://github.com/gnuradio/gr-inspector), Pandas (Python), Scikit (Python)

License: GPLv3 (Follows GNU Radio's licensing terms)

A naive kNN based automatic modulation classifier (features: BW, PSD_est(max), PSD_est(max) - PSD_est(min)).

* test.py: Validates the classifier by using 80% of the data from feature.csv to train and rest of them to test. Does not need GNU Radio to function.
* modrec.grc: A real-time modulation Recognition code, using GNU Radio.
* signal_separator.grc: A GNU Radio flow graph for the feature data collection.
* feature.csv: A sample feature data set.
* feature_handmeasured.csv: A sample feature data set.
