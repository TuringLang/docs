#! /bin/bash

# check to see if the markdown directory exists
if [ ! -d "./markdown" ]; then
    mkdir markdown
fi

for folder in tutorials/*; do
    #echo $folder
    rm $folder/Manifest.toml
    f="$(basename -- $folder)"
    echo $f
    julia -e "using TuringTutorials; TuringTutorials.weave_folder(\"$f\", (:github,))"
done

# exit 0
