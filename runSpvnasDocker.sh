

modelDirName="spvnas"
modelsBaseDir="/home/Documents"
modelDir="$modelsBaseDir/$modelDirName"
dataDirRoot="/home/Documents/data/tmp/dataset"
dataDir="/home/Documents/data/tmp/dataset/sequences/00/velodyne"
predDir="/home/Documents/data/out"

container="spvnas"
image=$container"_image"


modelRunCommand="torchpack dist-run " 
modelRunCommand+="-np 1 python3 evaluate.py configs/semantic_kitti/default.yaml "
modelRunCommand+="--name SemanticKITTI_val_SPVNAS@65GMACs "
modelRunCommand+="--data-dir $dataDirRoot/sequences "
modelRunCommand+="--save-dir $predDir"


# Run
# container name
# Access to GPUs
# User (otherwise we won't have permission to modify files created by bind mount)
# Mount model dir
# Mount data (bins) dir
# Mount predictions (bins) dir
# image
# bash (command) 


echo 
echo $modelRunCommand
echo 
echo Running Docker $container, $image 
echo 

docker run \
--name $container \
--gpus all \
--user "$(id -u)" \
--mount type=bind,source="$modelDir",target="$modelDir" \
--mount type=bind,source="$dataDir",target="$dataDir" \
--mount type=bind,source="$predDir",target="$predDir" \
$image \
bash -c "cd $modelDir && $modelRunCommand"



# Clean up container
docker container stop $container && docker container rm $container




