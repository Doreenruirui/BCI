source /home/dong.r/Library/Bash/bash_function.sh   
folder_data=$1
folder_train=$2
folder_out=$3
name_script=$4
dev=$5
size=128
machine=$6
jobname=$7
random=$8
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
echo $nline, $nfile
$(mkdir -p $folder_script)
for i in $(seq 1 $nfile);
do
    cur_file=$file_script$i
    j=$(($i-1))
    cur_start=$(($j*$size))
    cur_end=$(($i * $size))
    cur_cmd='python decode_seq2seq.py --random="'$random'" --model="seq2seq" --flag_bidirect=True --data_dir="'$folder_data'" --out_dir="'$folder_out'" --train_dir="'$folder_train'" --dev='$dev' --start='$cur_start' --end='$cur_end' --num_layers=3 --size=400 --num_cand=10 --num_pred=10 --beam_size=128 --batch_size=64 --num_wit=4'
    $(rm_file $cur_file)
    $(writejob $cur_file $jobname $i $folder_script $machine 'BCI')
    echo ''$cur_cmd >> $cur_file
    #$(sbatch $cur_file)
done
