sh run.sbatch.sub.train
sh run.sbatch.sub.dev
cd /home/dong.r/BCI/script/train_new
sbatch run.sbatch.lstm.sub.random.fix.69
sbatch run.sbatch.lstm.sub.random.fix.69.2
sbatch run.sbatch.lstm.sub.random.fix.69.3
sbatch run.sbatch.lstm.sub.random.fix.69.4
sbatch run.sbatch.lstm.sub.random.fix.69.5

