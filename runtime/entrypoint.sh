#!/bin/bash
set -euxo pipefail
exit_code=0

{
    cp -r /clouddata /data
    find /data -type d -exec chmod 700 {} \;
    find /data -type f -exec chmod 600 {} \;

    find /supervisor -type d -exec chmod 700 {} \;
    find /supervisor -type f -exec chmod 600 {} \;

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
		  python /supervisor/supervisor.py $prediction_time
	    sudo -u appuser \
		 /srv/conda/bin/conda run \
		 --no-capture-output \
		 -n nasa-airport-config-runtime \
		 python main.py $prediction_time

	    # Test that submission is valid
	    echo "Testing that prediction is valid"
	    conda run -n nasa-airport-config-runtime \
		  python /supervisor/scripts/check_prediction.py $prediction_time

	done < /data/prediction_times.txt

	echo "Constructing submission from individual predictions"
	conda run -n nasa-airport-config-runtime python /supervisor/scripts/construct_submission.py

	echo "Testing that submission is valid"
	conda run -n nasa-airport-config-runtime pytest /supervisor/scripts/test_submission.py

    else
        echo "ERROR: Could not find main.py in submission.zip"
        exit_code=1
    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
