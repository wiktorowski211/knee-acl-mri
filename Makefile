prepare_data:
	mkdir -p data/scans
	curl http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/metadata.csv --output data/metadata.csv
	for number in 01 02 03 04 05 06 07 08 09 10; do \
        path="http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/volumetric_data/vol$$number.7z"; \
		echo Downloading part $$number/10; \
		curl $$path --output /tmp/dataset.7z; \
		echo Downloaded... Extracting; \
		7z e /tmp/dataset.7z -odata/scans -aos; \
		echo Extracted... Cleaning; \
		rm /tmp/dataset.7z; \
    done
