source /home/dong.r/Library/Bash/bash_function.sh   
folder_data=$1
folder_train=$2
folder_out=$3
name_script=$4
dev=$5
size=1280
machine=$6
jobname=$7
random=$8
nbash=$9
folder_script='/home/dong.r/BCI/script/'$name_script
file_script=$folder_script'/run.sbatch.'
root_folder='/gss_gpfs_scratch/dong.r/Dataset/BCI/'
#root_folder='/home/dong.r/BCI/data/'
folder_data=$root_folder$folder_data
folder_out=$root_folder$folder_out
folder_train='/gss_gpfs_scratch/dong.r/Model/BCI/'$folder_train
#folder_train='/home/dong.r/BCI/model/'$folder_train
nline=$(cat $folder_data'/'$dev'.ids' | wc -l)
nfile=$(ceildiv $nline $size)
njob=$(ceildiv $nfile $nbash)
echo $nline, $nfile, $njob
$(mkdir -p $folder_script)
$(mkdir $folder_out)
for i in $(seq 1 $njob);
do  
    k=$(($i - 1))
    vsi=$(($k * $nbash))
    vsi=$(($vsi + 1))
    vei=$(($i * $nbash))
    vei=$(($nfile < $vei ? $nfile : $vei))
    echo $i, $vsi, $vei, $nbash
    cur_file=$file_script$i
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine 'BCI')
    bashcmd=''
    for j in $(seq $vsi $vei);
    do 
        echo $j,$vsi,$vei
        t=$((j - 1))
        vsj=$(($t * $size))
        vej=$(($j * $size))
        cur_cmd='python decode_seq2seq.py --random="'$random'" --model="lstm" --flag_bidirect=False --data_dir="'$folder_data'" --out_dir="'$folder_out'" --train_dir="'$folder_train'" --dev="'$dev'" --start='$vsj' --end='$vej' --num_layers=3 --size=400 --num_cand=10 --num_pred=10 --beam_size=128 --batch_size=128 --num_wit=4 --prior=3 --prob_high=0.7 --prob_noncand=0.0 --prob_in=1.0'
        bashfile=$folder_script'/bash.'$j
        $(rm_file $bashfile)
        echo ''$cur_cmd >> $bashfile
        if [ $j -ne $vei ];
        then
            bashcmd=$bashcmd'sh '$bashfile' & '
        else
            bashcmd=$bashcmd'sh '$bashfile
        fi
    done
    echo $bashcmd >> $cur_file 
done
