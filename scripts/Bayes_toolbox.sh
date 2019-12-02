#!/bin/bash
##################### Get file from Google Drive
gdown --id 1MHmTIqZunCPjrUG1sADqCMCoMGvuF7FQ --output ../data/Bayes_net_toolbox.rar
#####################remove the ^M(\r) in the name of generated files
#for file in `ls ../data/*.rar?`;do mv $file `echo $file|sed 's/\.rar\r/\.rar/g'`;done
