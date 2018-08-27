sh run.sbatch.sub.train.49
sh run.sbatch.sub.dev.49
cd /home/dong.r/BCI/script/train_new
sbatch run.sbatch.lstm.sub.random.fix.49
sbatch run.sbatch.lstm.sub.random.fix.49.2
sbatch run.sbatch.lstm.sub.random.fix.49.3
sbatch run.sbatch.lstm.sub.random.fix.49.4
sbatch run.sbatch.lstm.sub.random.fix.49.5
sbatch run.sbatch.s2s.sub.random.fix.49.5

