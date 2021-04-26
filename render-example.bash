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
    julia-1.5 -e "using Pkg; Pkg.instantiate(); using TuringTutorials; TuringTutorials.weave_folder(\"$f\", (:github,:script))" --project $f
done

# exit 0
