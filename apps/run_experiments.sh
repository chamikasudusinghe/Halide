#!/bin/bash

export HL_DEBUG_CODEGEN=1

# Number of invocation of Python Model during beam search
# iir_blur 45
# harris 154
# unsharp 189
# max_filter 324
# bgu 330
# interpolate 881
# local_laplacian 1541
# lens_blur 2024
# nl_means 2437

apps='
    iir_blur
    harris
    unsharp
    max_filter
    bgu
    interpolate
'
#    local_laplacian
#    lens_blur
#    nl_means
#'
#    --take out for now
#    stencil_chain
#    resnet_50
#    camera_pipe
#    bilateral_grid
#    conv_layer
#'

curloc=`pwd`;

num_cores=`nproc | cut -d' ' -f1`;

# Original version
for app in ${apps};
do
    while [[ 1 ]]; do
        running=$(jobs -r | wc -l)
	if [[ running -ge num_cores ]]; then
	    sleep 1
	else
	    break
	fi
    done
    
    exec_name="command.txt"

    cd ${curloc}/$app
    if [ $app == "harris" ] || [ $app == "iir_blur" ] || [ $app == "unsharp" ];
    then
        echo $app
        make clean all OPTIMIZE="-fno-rtti" HL_USE_TIRAMISU=0 &> original.txt
        ./${exec_name} >> original.txt 
        mv bin bin_original

        make clean all OPTIMIZE="-fno-rtti" HL_USE_TIRAMISU=1 &> new.txt &
    else
        make clean build OPTIMIZE="-fno-rtti" HL_USE_TIRAMISU=0 &> original.txt
        ./${exec_name} >> original.txt
        mv bin bin_original

        make clean build OPTIMIZE="-fno-rtti" HL_USE_TIRAMISU=1 &> compile_new.txt&
    fi
    echo -n .
done
wait
echo done.

#rm -rf tiramisu_*.txt *.json *_error.txt

