# Channel information

`scripts/fetch_channel_info.py` writes one `<dataset>.json` file per dataset into this directory (channel names, channel types, session and run counts of subject 1). `scripts/run_evaluation.py` reads the channel names from these files to form the sensorimotor montage configuration (`SM`); without them, `SM` is skipped.
