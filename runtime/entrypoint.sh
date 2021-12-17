#!/bin/bash
set -euxo pipefail
exit_code=0

{
    cp -r /clouddata /data
    find /data -type d -exec chmod 700 {} \;
    find /data -type f -exec chmod 600 {} \;

    cd /codeexecution

    echo "List installed packages"
    echo "######################################"
    conda list -n nasa-airport-config-runtime
    echo "######################################"

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./
    ls -alh

    if [ -f "main.py" ]
    then
        echo "Running submission with Python"
	cp /data/aircraft_types_mapping.csv /codeexecution/data/aircraft_types_mapping.csv

	while read prediction_time
	do
	    conda run \
		  --no-capture-output \
		  -n nasa-airport-config-runtime \
		  python supervisor.py $prediction_time
	    sudo -u appuser \
		 /srv/conda/bin/conda run \
		 --no-capture-output \
		 -n nasa-airport-config-runtime \
		 python main.py $prediction_time
	done < /data/prediction_times.txt

	# Test that submission is valid
	echo "Testing that submission is valid"
	conda run -n nasa-airport-config-runtime pytest -v tests/test_submission.py

	echo "Compressing files in a gzipped tar archive for submission"
	cd ./submission \
	  && tar czf ./submission.tar.gz *.tif \
	  && rm ./*.tif \
	  && cd ..

	echo "... finished"
	du -h submission/submission.tar.gz

    else
        echo "ERROR: Could not find main.py in submission.zip"
        exit_code=1
    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
