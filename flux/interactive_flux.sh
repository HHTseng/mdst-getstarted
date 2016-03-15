#!/bin/bash
qsub \
-I \
-N interactive_job \
-M uniqname@umich.edu \
-m abe \
-A mdatascienceteam_flux \
-q flux \
-l qos=flux,nodes=1:ppn=1,pmem=4gb,walltime=01:00:00:00 \
-j oe \
-V \
-d "/scratch/mdatascienceteam_flux/uniqname/"
