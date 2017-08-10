#!/bin/bash


#
for dataset in  prid viper 3dpes ilids cuhk01 cuhk03; 
do 
    datadir="/home/nikoong/dataset/re-id/"$dataset
    cam_0=$datadir"/cam_0"
    cam_1=$datadir"/cam_1"

    posedatadir="/home/nikoong/dataset/re-id_pose/"$dataset
    renderdir=$posedatadir"/rendered"
    render0=$renderdir"/cam_0"
    render1=$renderdir"/cam_1"
    jsondir=$posedatadir"/json"
    json_0=$jsondir"/cam_0"
    json_1=$jsondir"/cam_1"
    updir=$posedatadir"/up"
    downdir=$posedatadir"/down"    
    #rm -r ${jsondir}
    mkdir $posedatadir
    mkdir $renderdir
    mkdir $render0
    mkdir $render1
    mkdir $jsondir
    mkdir $json_0
    mkdir $json_1 
    mkdir $updir
    mkdir $updir"/cam_0"
    mkdir $updir"/cam_1"
    mkdir $downdir
    mkdir $downdir"/cam_0"
    mkdir $downdir"/cam_1"  
    /home/nikoong/Algorithm_test/openpose/build/examples/openpose/openpose.bin -image_dir $cam_0  -write_pose_json $json_0 -scale_mode 0 -no_display ture -write_images $render0
    /home/nikoong/Algorithm_test/openpose/build/examples/openpose/openpose.bin -image_dir $cam_1  -write_pose_json $json_1 -scale_mode 0 -no_display ture $render1
     
done
