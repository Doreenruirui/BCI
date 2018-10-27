I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
INFO:root:Get NLC data in /gss_gpfs_scratch/dong.r/Dataset/BCI/Sublex/1/1/train/sub
INFO:root:Vocabulary size: 30
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties: 
name: Tesla K20m
major: 3 minor: 5 memoryClockRate (GHz) 0.7055
pciBusID 0000:04:00.0
Total memory: 4.63GiB
Free memory: 4.56GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:838] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K20m, pci bus id: 0000:04:00.0)
INFO:root:Creating 3 layers of 512 units.
INFO:root:Reading model parameters from /gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-38
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 3033 get requests, put_count=2372 evicted_count=1000 eviction_rate=0.421585 and unsatisfied allocation rate=0.580613
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 3178 get requests, put_count=2277 evicted_count=1000 eviction_rate=0.439174 and unsatisfied allocation rate=0.604154
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 212 to 233
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 237 get requests, put_count=1270 evicted_count=1000 eviction_rate=0.787402 and unsatisfied allocation rate=0
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 3921 get requests, put_count=3146 evicted_count=1000 eviction_rate=0.317864 and unsatisfied allocation rate=0.465187
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 542 to 596
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 5544 get requests, put_count=4644 evicted_count=1000 eviction_rate=0.215332 and unsatisfied allocation rate=0.363456
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 1273 to 1400
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 19572 get requests, put_count=18848 evicted_count=1000 eviction_rate=0.053056 and unsatisfied allocation rate=0.099581
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 2478 to 2725
INFO:root:epoch 39, iter 200, cost 2.034126, exp_cost 1.593577, grad norm 2.890177, param norm 1377.519287, tps 413.597946, length mean/std 3.000000/0.000000
INFO:root:epoch 39, iter 400, cost 1.689011, exp_cost 1.595726, grad norm 3.555846, param norm 1377.848511, tps 585.024043, length mean/std 4.000000/0.000000
INFO:root:epoch 39, iter 600, cost 1.534389, exp_cost 1.595940, grad norm 7.094007, param norm 1378.094238, tps 680.417670, length mean/std 11.000000/0.000000
INFO:root:epoch 39, iter 800, cost 1.844468, exp_cost 1.594791, grad norm 2.919025, param norm 1378.445312, tps 738.628703, length mean/std 3.000000/0.000000
INFO:root:epoch 39, iter 1000, cost 1.498910, exp_cost 1.588243, grad norm 6.618623, param norm 1378.744995, tps 782.559260, length mean/std 10.000000/0.000000
INFO:root:epoch 39, iter 1200, cost 1.512226, exp_cost 1.583685, grad norm 7.564834, param norm 1379.012939, tps 816.709164, length mean/std 11.000000/0.000000
INFO:root:epoch 39, iter 1400, cost 2.110182, exp_cost 1.603500, grad norm 1.999046, param norm 1379.327148, tps 837.730870, length mean/std 2.000000/0.000000
INFO:root:epoch 39, iter 1600, cost 1.609916, exp_cost 1.588391, grad norm 12.385878, param norm 1379.744141, tps 856.672824, length mean/std 19.000000/0.000000
INFO:root:epoch 39, iter 1800, cost 1.538529, exp_cost 1.586928, grad norm 16.827040, param norm 1380.006836, tps 871.223234, length mean/std 28.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 6409458 get requests, put_count=6409666 evicted_count=4000 eviction_rate=0.000624057 and unsatisfied allocation rate=0.000659962
INFO:root:epoch 39, iter 2000, cost 2.259257, exp_cost 1.594444, grad norm 2.055142, param norm 1380.332275, tps 885.086112, length mean/std 2.000000/0.000000
INFO:root:epoch 39, iter 2200, cost 1.477378, exp_cost 1.584757, grad norm 9.633524, param norm 1380.606079, tps 895.673741, length mean/std 15.000000/0.000000
INFO:root:epoch 39, iter 2400, cost 1.780041, exp_cost 1.596415, grad norm 4.717057, param norm 1380.937012, tps 904.663288, length mean/std 6.000000/0.000000
INFO:root:epoch 39, iter 2600, cost 2.847640, exp_cost 1.601828, grad norm 0.631520, param norm 1381.184570, tps 912.879464, length mean/std 1.000000/0.000000
INFO:root:epoch 39, iter 2800, cost 1.575198, exp_cost 1.592382, grad norm 4.309774, param norm 1381.549072, tps 919.318936, length mean/std 6.000000/0.000000
INFO:root:epoch 39, iter 3000, cost 1.528731, exp_cost 1.601461, grad norm 11.438546, param norm 1381.857544, tps 925.447409, length mean/std 17.000000/0.000000
INFO:root:epoch 39, iter 3200, cost 1.585673, exp_cost 1.591035, grad norm 20.522079, param norm 1382.220825, tps 930.261270, length mean/std 31.000000/0.000000
INFO:root:epoch 39, iter 3400, cost 3.153183, exp_cost 1.600145, grad norm 0.800220, param norm 1382.477661, tps 935.679087, length mean/std 1.000000/0.000000
INFO:root:epoch 39, iter 3600, cost 1.475453, exp_cost 1.604546, grad norm 13.237410, param norm 1382.813843, tps 940.120406, length mean/std 23.000000/0.000000
INFO:root:epoch 39, iter 3800, cost 2.200102, exp_cost 1.604724, grad norm 1.627311, param norm 1383.165894, tps 944.267584, length mean/std 2.000000/0.000000
INFO:root:epoch 39, iter 4000, cost 1.551612, exp_cost 1.593632, grad norm 13.426826, param norm 1383.414307, tps 948.002514, length mean/std 20.000000/0.000000
INFO:root:epoch 39, iter 4200, cost 2.028241, exp_cost 1.592822, grad norm 2.580681, param norm 1383.719116, tps 951.125670, length mean/std 3.000000/0.000000
INFO:root:epoch 39, iter 4400, cost 1.439204, exp_cost 1.584558, grad norm 8.271132, param norm 1383.950073, tps 954.062011, length mean/std 13.000000/0.000000
INFO:root:epoch 39, iter 4600, cost 1.568606, exp_cost 1.598785, grad norm 15.935152, param norm 1384.299194, tps 956.729085, length mean/std 25.000000/0.000000
INFO:root:epoch 39, iter 4800, cost 1.584953, exp_cost 1.596463, grad norm 19.104832, param norm 1384.610352, tps 959.149023, length mean/std 32.000000/0.000000
INFO:root:epoch 39, iter 5000, cost 1.504105, exp_cost 1.594000, grad norm 11.366659, param norm 1384.924438, tps 961.730331, length mean/std 18.000000/0.000000
INFO:root:epoch 39, iter 5200, cost 1.417417, exp_cost 1.593470, grad norm 10.674301, param norm 1385.197632, tps 964.587244, length mean/std 18.000000/0.000000
INFO:root:epoch 39, iter 5400, cost 1.580381, exp_cost 1.600858, grad norm 19.895683, param norm 1385.464478, tps 966.554352, length mean/std 29.000000/0.000000
INFO:root:epoch 39, iter 5600, cost 1.607072, exp_cost 1.599211, grad norm 19.693405, param norm 1385.761353, tps 968.460275, length mean/std 32.000000/0.000000
INFO:root:epoch 39, iter 5800, cost 1.869970, exp_cost 1.599741, grad norm 2.723337, param norm 1386.062012, tps 969.923559, length mean/std 3.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 22049276 get requests, put_count=22049620 evicted_count=14000 eviction_rate=0.000634932 and unsatisfied allocation rate=0.000639205
INFO:root:epoch 39, iter 6000, cost 2.079918, exp_cost 1.599821, grad norm 2.664281, param norm 1386.418945, tps 971.529034, length mean/std 3.000000/0.000000
INFO:root:epoch 39, iter 6200, cost 1.701602, exp_cost 1.599993, grad norm 24.165436, param norm 1386.726196, tps 973.115093, length mean/std 38.000000/0.000000
INFO:root:epoch 39, iter 6400, cost 1.672167, exp_cost 1.603275, grad norm 22.025389, param norm 1387.102661, tps 974.577624, length mean/std 36.000000/0.000000
INFO:root:epoch 39, iter 6600, cost 1.581024, exp_cost 1.613909, grad norm 9.751487, param norm 1387.354004, tps 976.076847, length mean/std 14.000000/0.000000
INFO:root:epoch 39, iter 6800, cost 1.628873, exp_cost 1.599993, grad norm 3.038626, param norm 1387.699341, tps 977.637842, length mean/std 4.000000/0.000000
INFO:root:epoch 39, iter 7000, cost 1.597619, exp_cost 1.605377, grad norm 14.163680, param norm 1387.961304, tps 978.780818, length mean/std 23.000000/0.000000
INFO:root:epoch 39, iter 7200, cost 1.608845, exp_cost 1.599723, grad norm 14.348503, param norm 1388.294312, tps 979.879784, length mean/std 23.000000/0.000000
INFO:root:epoch 39, iter 7400, cost 1.578792, exp_cost 1.595283, grad norm 12.155499, param norm 1388.598755, tps 980.887294, length mean/std 20.000000/0.000000
INFO:root:epoch 39, iter 7600, cost 1.688986, exp_cost 1.608145, grad norm 12.369146, param norm 1388.948608, tps 981.686078, length mean/std 18.000000/0.000000
INFO:root:epoch 39, iter 7800, cost 1.856288, exp_cost 1.613254, grad norm 3.425627, param norm 1389.229248, tps 982.780200, length mean/std 4.000000/0.000000
INFO:root:epoch 39, iter 8000, cost 1.648101, exp_cost 1.598716, grad norm 12.224791, param norm 1389.472046, tps 983.707956, length mean/std 19.000000/0.000000
INFO:root:epoch 39, iter 8200, cost 1.412658, exp_cost 1.597200, grad norm 9.840983, param norm 1389.764038, tps 984.622057, length mean/std 16.000000/0.000000
INFO:root:epoch 39, iter 8400, cost 1.640015, exp_cost 1.602955, grad norm 19.085230, param norm 1390.071899, tps 985.477910, length mean/std 30.000000/0.000000
INFO:root:epoch 39, iter 8600, cost 2.786915, exp_cost 1.594820, grad norm 0.533415, param norm 1390.291260, tps 986.258153, length mean/std 1.000000/0.000000
INFO:root:epoch 39, iter 8800, cost 1.703902, exp_cost 1.606293, grad norm 3.289889, param norm 1390.637573, tps 987.163950, length mean/std 3.875000/0.330719
INFO:root:epoch 39, iter 9000, cost 1.596490, exp_cost 1.607391, grad norm 20.137772, param norm 1390.960327, tps 988.019395, length mean/std 33.000000/0.000000
INFO:root:epoch 39, iter 9200, cost 1.664586, exp_cost 1.610492, grad norm 10.333732, param norm 1391.225098, tps 988.653918, length mean/std 15.000000/0.000000
INFO:root:epoch 39, iter 9400, cost 1.521553, exp_cost 1.615040, grad norm 10.409836, param norm 1391.518311, tps 989.416676, length mean/std 16.000000/0.000000
INFO:root:epoch 39, iter 9600, cost 2.208046, exp_cost 1.604927, grad norm 1.912935, param norm 1391.820679, tps 990.083044, length mean/std 2.000000/0.000000
INFO:root:epoch 39, iter 9800, cost 1.528308, exp_cost 1.606106, grad norm 11.578974, param norm 1392.131836, tps 990.711260, length mean/std 18.000000/0.000000
INFO:root:epoch 39, iter 10000, cost 2.937220, exp_cost 1.597118, grad norm 0.666182, param norm 1392.457153, tps 991.242085, length mean/std 1.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 38171371 get requests, put_count=38171370 evicted_count=24000 eviction_rate=0.000628743 and unsatisfied allocation rate=0.000640244
INFO:root:epoch 39, iter 10200, cost 1.578124, exp_cost 1.603413, grad norm 12.812981, param norm 1392.773315, tps 991.927541, length mean/std 19.000000/0.000000
INFO:root:epoch 39, iter 10400, cost 1.653298, exp_cost 1.617592, grad norm 21.706078, param norm 1393.082520, tps 992.366156, length mean/std 39.000000/0.000000
INFO:root:epoch 39, iter 10600, cost 1.563909, exp_cost 1.607369, grad norm 14.436767, param norm 1393.357422, tps 992.999634, length mean/std 22.000000/0.000000
INFO:root:epoch 39, iter 10800, cost 1.639372, exp_cost 1.610084, grad norm 10.588611, param norm 1393.609375, tps 993.728800, length mean/std 16.000000/0.000000
INFO:root:epoch 39, iter 11000, cost 1.608526, exp_cost 1.617485, grad norm 5.464077, param norm 1393.928345, tps 994.269323, length mean/std 8.000000/0.000000
INFO:root:epoch 39, iter 11200, cost 1.559880, exp_cost 1.611206, grad norm 12.753537, param norm 1394.244751, tps 994.709484, length mean/std 20.000000/0.000000
INFO:root:epoch 39, iter 11400, cost 1.600987, exp_cost 1.607614, grad norm 11.421419, param norm 1394.575562, tps 995.003383, length mean/std 17.000000/0.000000
INFO:root:epoch 39, iter 11600, cost 1.567492, exp_cost 1.607104, grad norm 13.904822, param norm 1394.814819, tps 995.427465, length mean/std 23.000000/0.000000
INFO:root:epoch 39, iter 11800, cost 1.455332, exp_cost 1.597108, grad norm 8.071652, param norm 1395.123291, tps 995.976306, length mean/std 13.000000/0.000000
INFO:root:epoch 39, iter 12000, cost 1.743819, exp_cost 1.610727, grad norm 7.335676, param norm 1395.556885, tps 996.510010, length mean/std 10.000000/0.000000
INFO:root:epoch 39, iter 12200, cost 1.529470, exp_cost 1.601571, grad norm 11.575943, param norm 1395.734253, tps 996.770731, length mean/std 18.000000/0.000000
INFO:root:epoch 39, iter 12400, cost 1.698332, exp_cost 1.614958, grad norm 4.660760, param norm 1396.034912, tps 997.107105, length mean/std 6.000000/0.000000
INFO:root:Epoch 39 Validation cost: 1.702688 time: 6488.014370
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-39 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-39 is not in all_model_checkpoint_paths. Manually adding it.
INFO:root:epoch 40, iter 200, cost 1.471330, exp_cost 1.596777, grad norm 5.424208, param norm 1396.506592, tps 906.861375, length mean/std 8.000000/0.000000
INFO:root:epoch 40, iter 400, cost 1.517639, exp_cost 1.586645, grad norm 12.086272, param norm 1396.831909, tps 908.479991, length mean/std 19.000000/0.000000
INFO:root:epoch 40, iter 600, cost 1.680446, exp_cost 1.584853, grad norm 20.923714, param norm 1397.060181, tps 910.088503, length mean/std 34.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 51653545 get requests, put_count=51653880 evicted_count=34000 eviction_rate=0.000658227 and unsatisfied allocation rate=0.000660226
INFO:root:epoch 40, iter 800, cost 1.586747, exp_cost 1.584927, grad norm 15.998997, param norm 1397.327148, tps 911.450061, length mean/std 25.000000/0.000000
INFO:root:epoch 40, iter 1000, cost 1.634347, exp_cost 1.584346, grad norm 9.858969, param norm 1397.660889, tps 912.872853, length mean/std 14.000000/0.000000
INFO:root:epoch 40, iter 1200, cost 1.512652, exp_cost 1.596624, grad norm 11.282743, param norm 1397.958252, tps 914.420404, length mean/std 17.000000/0.000000
INFO:root:epoch 40, iter 1400, cost 1.477792, exp_cost 1.595842, grad norm 13.176171, param norm 1398.279785, tps 915.907893, length mean/std 21.000000/0.000000
INFO:root:epoch 40, iter 1600, cost 1.557265, exp_cost 1.599203, grad norm 10.974674, param norm 1398.584351, tps 917.296138, length mean/std 19.000000/0.000000
INFO:root:epoch 40, iter 1800, cost 1.430952, exp_cost 1.587336, grad norm 11.593612, param norm 1398.911621, tps 918.553240, length mean/std 20.000000/0.000000
INFO:root:epoch 40, iter 2000, cost 1.576009, exp_cost 1.585884, grad norm 8.488305, param norm 1399.275879, tps 919.923514, length mean/std 12.000000/0.000000
INFO:root:epoch 40, iter 2200, cost 1.639245, exp_cost 1.585593, grad norm 16.430704, param norm 1399.575195, tps 921.256416, length mean/std 27.000000/0.000000
INFO:root:epoch 40, iter 2400, cost 1.574796, exp_cost 1.596503, grad norm 20.877289, param norm 1399.931030, tps 922.695620, length mean/std 35.000000/0.000000
INFO:root:epoch 40, iter 2600, cost 1.739190, exp_cost 1.589445, grad norm 4.711029, param norm 1400.263916, tps 923.821299, length mean/std 6.000000/0.000000
INFO:root:epoch 40, iter 2800, cost 1.590039, exp_cost 1.602496, grad norm 7.861225, param norm 1400.565063, tps 924.972824, length mean/std 11.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 59873087 get requests, put_count=59873306 evicted_count=44000 eviction_rate=0.000734885 and unsatisfied allocation rate=0.000738546
INFO:root:epoch 40, iter 3000, cost 1.571126, exp_cost 1.586459, grad norm 21.513351, param norm 1400.842041, tps 926.317620, length mean/std 37.000000/0.000000
INFO:root:epoch 40, iter 3200, cost 1.558521, exp_cost 1.590068, grad norm 6.177736, param norm 1401.154907, tps 927.383523, length mean/std 9.000000/0.000000
INFO:root:epoch 40, iter 3400, cost 1.513691, exp_cost 1.590631, grad norm 17.248075, param norm 1401.455444, tps 928.550223, length mean/std 28.000000/0.000000
INFO:root:epoch 40, iter 3600, cost 1.575743, exp_cost 1.596081, grad norm 10.213750, param norm 1401.749878, tps 929.722497, length mean/std 15.000000/0.000000
INFO:root:epoch 40, iter 3800, cost 1.580945, exp_cost 1.613308, grad norm 12.692698, param norm 1402.104248, tps 930.798606, length mean/std 20.000000/0.000000
INFO:root:epoch 40, iter 4000, cost 2.016839, exp_cost 1.593847, grad norm 2.934858, param norm 1402.304810, tps 931.742601, length mean/std 3.000000/0.000000
INFO:root:epoch 40, iter 4200, cost 1.566431, exp_cost 1.595010, grad norm 16.169115, param norm 1402.559814, tps 932.722576, length mean/std 28.000000/0.000000
INFO:root:epoch 40, iter 4400, cost 1.575451, exp_cost 1.598093, grad norm 8.730732, param norm 1402.823608, tps 933.816214, length mean/std 12.000000/0.000000
INFO:root:epoch 40, iter 4600, cost 1.576314, exp_cost 1.600778, grad norm 10.628879, param norm 1403.090820, tps 934.743339, length mean/std 17.000000/0.000000
INFO:root:epoch 40, iter 4800, cost 1.442195, exp_cost 1.609558, grad norm 8.812357, param norm 1403.396973, tps 935.665040, length mean/std 13.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 67918284 get requests, put_count=67918497 evicted_count=54000 eviction_rate=0.000795071 and unsatisfied allocation rate=0.000798386
INFO:root:epoch 40, iter 5000, cost 1.717030, exp_cost 1.599493, grad norm 3.263137, param norm 1403.600464, tps 936.508752, length mean/std 4.000000/0.000000
INFO:root:epoch 40, iter 5200, cost 1.687873, exp_cost 1.606620, grad norm 18.656652, param norm 1403.864136, tps 937.403543, length mean/std 30.000000/0.000000
INFO:root:epoch 40, iter 5400, cost 1.683665, exp_cost 1.610620, grad norm 4.327055, param norm 1404.186768, tps 938.346471, length mean/std 5.000000/0.000000
INFO:root:epoch 40, iter 5600, cost 1.884737, exp_cost 1.609518, grad norm 3.268004, param norm 1404.499878, tps 939.187593, length mean/std 4.000000/0.000000
INFO:root:epoch 40, iter 5800, cost 1.455901, exp_cost 1.597951, grad norm 6.548470, param norm 1404.799438, tps 940.068422, length mean/std 10.000000/0.000000
INFO:root:epoch 40, iter 6000, cost 1.471226, exp_cost 1.601595, grad norm 6.329424, param norm 1405.143555, tps 940.848831, length mean/std 9.000000/0.000000
INFO:root:epoch 40, iter 6200, cost 1.547595, exp_cost 1.600277, grad norm 7.684228, param norm 1405.406494, tps 941.681938, length mean/std 12.000000/0.000000
INFO:root:epoch 40, iter 6400, cost 1.502704, exp_cost 1.591105, grad norm 8.066876, param norm 1405.676147, tps 942.461201, length mean/std 12.000000/0.000000
INFO:root:epoch 40, iter 6600, cost 2.133377, exp_cost 1.594813, grad norm 1.759915, param norm 1406.059448, tps 943.200030, length mean/std 2.000000/0.000000
INFO:root:epoch 40, iter 6800, cost 1.508892, exp_cost 1.590124, grad norm 13.489870, param norm 1406.396606, tps 944.020826, length mean/std 21.000000/0.000000
INFO:root:epoch 40, iter 7000, cost 2.015689, exp_cost 1.599922, grad norm 2.758999, param norm 1406.685425, tps 944.704510, length mean/std 3.000000/0.000000
INFO:root:epoch 40, iter 7200, cost 1.523973, exp_cost 1.597466, grad norm 11.179988, param norm 1407.057617, tps 945.479925, length mean/std 17.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 77019703 get requests, put_count=77020057 evicted_count=64000 eviction_rate=0.000830952 and unsatisfied allocation rate=0.000832047
INFO:root:epoch 40, iter 7400, cost 1.525362, exp_cost 1.604180, grad norm 7.429685, param norm 1407.283203, tps 946.253477, length mean/std 11.000000/0.000000
INFO:root:epoch 40, iter 7600, cost 1.578124, exp_cost 1.597339, grad norm 3.711599, param norm 1407.588257, tps 946.907541, length mean/std 5.000000/0.000000
INFO:root:epoch 40, iter 7800, cost 1.511916, exp_cost 1.604467, grad norm 12.319413, param norm 1407.812988, tps 947.698270, length mean/std 19.000000/0.000000
INFO:root:epoch 40, iter 8000, cost 1.568400, exp_cost 1.597358, grad norm 14.210093, param norm 1408.060913, tps 948.550545, length mean/std 21.000000/0.000000
INFO:root:epoch 40, iter 8200, cost 1.509844, exp_cost 1.594273, grad norm 12.617625, param norm 1408.347778, tps 949.238624, length mean/std 21.000000/0.000000
INFO:root:epoch 40, iter 8400, cost 1.542756, exp_cost 1.608910, grad norm 15.973307, param norm 1408.730713, tps 949.949719, length mean/std 26.000000/0.000000
INFO:root:epoch 40, iter 8600, cost 1.554126, exp_cost 1.595527, grad norm 7.068621, param norm 1408.988281, tps 950.618572, length mean/std 10.000000/0.000000
INFO:root:epoch 40, iter 8800, cost 1.584115, exp_cost 1.600439, grad norm 16.335760, param norm 1409.256104, tps 951.106242, length mean/std 26.000000/0.000000
INFO:root:epoch 40, iter 9000, cost 1.689576, exp_cost 1.594016, grad norm 9.106101, param norm 1409.594727, tps 951.808588, length mean/std 12.000000/0.000000
INFO:root:epoch 40, iter 9200, cost 1.719060, exp_cost 1.604608, grad norm 5.869017, param norm 1409.977051, tps 952.422503, length mean/std 8.000000/0.000000
INFO:root:epoch 40, iter 9400, cost 1.670294, exp_cost 1.599634, grad norm 3.763185, param norm 1410.335083, tps 953.072382, length mean/std 5.000000/0.000000
INFO:root:epoch 40, iter 9600, cost 1.541746, exp_cost 1.599032, grad norm 14.357997, param norm 1410.625610, tps 953.692809, length mean/std 23.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 86227051 get requests, put_count=86227294 evicted_count=74000 eviction_rate=0.000858197 and unsatisfied allocation rate=0.000860461
INFO:root:epoch 40, iter 9800, cost 1.579830, exp_cost 1.600720, grad norm 3.111169, param norm 1410.912354, tps 954.289508, length mean/std 4.000000/0.000000
INFO:root:epoch 40, iter 10000, cost 1.469722, exp_cost 1.611413, grad norm 7.405995, param norm 1411.293457, tps 954.902200, length mean/std 11.000000/0.000000
INFO:root:epoch 40, iter 10200, cost 1.462749, exp_cost 1.595131, grad norm 9.566754, param norm 1411.490479, tps 955.460640, length mean/std 16.000000/0.000000
INFO:root:epoch 40, iter 10400, cost 2.213073, exp_cost 1.606986, grad norm 1.684498, param norm 1411.864014, tps 956.013273, length mean/std 2.000000/0.000000
INFO:root:epoch 40, iter 10600, cost 1.600472, exp_cost 1.600377, grad norm 7.766040, param norm 1412.211182, tps 956.479374, length mean/std 11.000000/0.000000
INFO:root:epoch 40, iter 10800, cost 1.447904, exp_cost 1.602156, grad norm 12.326863, param norm 1412.476440, tps 956.997345, length mean/std 21.000000/0.000000
INFO:root:epoch 40, iter 11000, cost 1.610044, exp_cost 1.599606, grad norm 16.990093, param norm 1412.815552, tps 957.701464, length mean/std 27.000000/0.000000
INFO:root:epoch 40, iter 11200, cost 1.586326, exp_cost 1.603887, grad norm 23.265404, param norm 1413.180908, tps 958.195706, length mean/std 35.000000/0.000000
INFO:root:epoch 40, iter 11400, cost 1.661104, exp_cost 1.607985, grad norm 20.821714, param norm 1413.404907, tps 958.702044, length mean/std 35.000000/0.000000
INFO:root:epoch 40, iter 11600, cost 1.603698, exp_cost 1.609617, grad norm 13.797900, param norm 1413.804565, tps 959.238730, length mean/std 21.000000/0.000000
INFO:root:epoch 40, iter 11800, cost 1.664183, exp_cost 1.608139, grad norm 21.744820, param norm 1414.028442, tps 959.672902, length mean/std 36.000000/0.000000
INFO:root:epoch 40, iter 12000, cost 1.686868, exp_cost 1.615491, grad norm 22.536184, param norm 1414.315918, tps 960.092774, length mean/std 36.000000/0.000000
INFO:root:epoch 40, iter 12200, cost 1.539827, exp_cost 1.608087, grad norm 11.614182, param norm 1414.587402, tps 960.588807, length mean/std 18.000000/0.000000
INFO:root:epoch 40, iter 12400, cost 1.740158, exp_cost 1.602595, grad norm 7.778623, param norm 1414.841919, tps 961.050197, length mean/std 10.000000/0.000000
INFO:root:Epoch 40 Validation cost: 1.704451 time: 6565.013111
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-40 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-40 is not in all_model_checkpoint_paths. Manually adding it.
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 98649832 get requests, put_count=98650041 evicted_count=84000 eviction_rate=0.000851495 and unsatisfied allocation rate=0.000853818
INFO:root:epoch 41, iter 200, cost 1.501013, exp_cost 1.591362, grad norm 6.948640, param norm 1415.296753, tps 917.193871, length mean/std 11.000000/0.000000
INFO:root:epoch 41, iter 400, cost 1.517291, exp_cost 1.586190, grad norm 9.263371, param norm 1415.594360, tps 918.084814, length mean/std 13.000000/0.000000
INFO:root:epoch 41, iter 600, cost 1.513062, exp_cost 1.584333, grad norm 12.887459, param norm 1415.914185, tps 918.880693, length mean/std 21.000000/0.000000
INFO:root:epoch 41, iter 800, cost 1.517002, exp_cost 1.583251, grad norm 12.693873, param norm 1416.194580, tps 919.659825, length mean/std 21.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 102214443 get requests, put_count=102214606 evicted_count=94000 eviction_rate=0.000919634 and unsatisfied allocation rate=0.000922326
INFO:root:epoch 41, iter 1000, cost 1.495257, exp_cost 1.582961, grad norm 14.500131, param norm 1416.540771, tps 920.461021, length mean/std 23.000000/0.000000
INFO:root:epoch 41, iter 1200, cost 1.514658, exp_cost 1.585261, grad norm 8.443147, param norm 1416.853149, tps 921.201003, length mean/std 13.000000/0.000000
INFO:root:epoch 41, iter 1400, cost 1.556648, exp_cost 1.569318, grad norm 20.538881, param norm 1417.181519, tps 921.820103, length mean/std 36.000000/0.000000
INFO:root:epoch 41, iter 1600, cost 1.650979, exp_cost 1.591415, grad norm 3.651893, param norm 1417.492432, tps 922.653408, length mean/std 4.000000/0.000000
INFO:root:epoch 41, iter 1800, cost 1.597516, exp_cost 1.595308, grad norm 11.239784, param norm 1417.756226, tps 923.326583, length mean/std 17.000000/0.000000
INFO:root:epoch 41, iter 2000, cost 1.622494, exp_cost 1.596202, grad norm 17.144167, param norm 1418.113647, tps 924.039651, length mean/std 28.000000/0.000000
INFO:root:epoch 41, iter 2200, cost 1.570658, exp_cost 1.591246, grad norm 23.276833, param norm 1418.457031, tps 924.756421, length mean/std 41.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 107678674 get requests, put_count=107678968 evicted_count=104000 eviction_rate=0.000965834 and unsatisfied allocation rate=0.000967174
INFO:root:epoch 41, iter 2400, cost 1.469701, exp_cost 1.584116, grad norm 6.081113, param norm 1418.643311, tps 925.458778, length mean/std 9.000000/0.000000
INFO:root:epoch 41, iter 2600, cost 1.463913, exp_cost 1.580170, grad norm 10.629129, param norm 1418.957275, tps 926.092699, length mean/std 18.000000/0.000000
INFO:root:epoch 41, iter 2800, cost 1.630862, exp_cost 1.591272, grad norm 19.342276, param norm 1419.277832, tps 926.769484, length mean/std 32.000000/0.000000
INFO:root:epoch 41, iter 3000, cost 1.518215, exp_cost 1.588041, grad norm 15.312596, param norm 1419.566162, tps 927.409566, length mean/std 23.000000/0.000000
INFO:root:epoch 41, iter 3200, cost 1.637291, exp_cost 1.593558, grad norm 8.475327, param norm 1419.895508, tps 927.973754, length mean/std 12.000000/0.000000
INFO:root:epoch 41, iter 3400, cost 1.563586, exp_cost 1.598906, grad norm 15.834280, param norm 1420.123657, tps 928.665039, length mean/std 25.000000/0.000000
INFO:root:epoch 41, iter 3600, cost 1.949185, exp_cost 1.592421, grad norm 2.990926, param norm 1420.379517, tps 929.266986, length mean/std 3.000000/0.000000
INFO:root:epoch 41, iter 3800, cost 1.633229, exp_cost 1.598548, grad norm 3.871317, param norm 1420.757202, tps 929.878522, length mean/std 5.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 113622962 get requests, put_count=113623238 evicted_count=114000 eviction_rate=0.00100332 and unsatisfied allocation rate=0.00100474
INFO:root:epoch 41, iter 4000, cost 1.762725, exp_cost 1.599215, grad norm 4.853865, param norm 1421.040161, tps 930.531264, length mean/std 6.000000/0.000000
INFO:root:epoch 41, iter 4200, cost 1.896739, exp_cost 1.594283, grad norm 2.445885, param norm 1421.302002, tps 931.107318, length mean/std 3.000000/0.000000
INFO:root:epoch 41, iter 4400, cost 1.579734, exp_cost 1.594155, grad norm 15.819312, param norm 1421.538818, tps 931.739875, length mean/std 23.000000/0.000000
INFO:root:epoch 41, iter 4600, cost 1.755544, exp_cost 1.595103, grad norm 16.993694, param norm 1421.881836, tps 932.330613, length mean/std 21.000000/0.000000
INFO:root:epoch 41, iter 4800, cost 1.485295, exp_cost 1.580925, grad norm 10.919042, param norm 1422.150146, tps 932.919885, length mean/std 17.000000/0.000000
INFO:root:epoch 41, iter 5000, cost 1.597342, exp_cost 1.584659, grad norm 15.647845, param norm 1422.452393, tps 933.524565, length mean/std 24.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 119075867 get requests, put_count=119075950 evicted_count=124000 eviction_rate=0.00104135 and unsatisfied allocation rate=0.00104433
INFO:root:epoch 41, iter 5200, cost 1.496017, exp_cost 1.595904, grad norm 8.698545, param norm 1422.682007, tps 934.033861, length mean/std 12.000000/0.000000
INFO:root:epoch 41, iter 5400, cost 1.638165, exp_cost 1.601544, grad norm 4.539319, param norm 1422.996826, tps 934.608718, length mean/std 6.000000/0.000000
INFO:root:epoch 41, iter 5600, cost 1.574520, exp_cost 1.599350, grad norm 18.387209, param norm 1423.315796, tps 935.120888, length mean/std 30.000000/0.000000
INFO:root:epoch 41, iter 5800, cost 1.876300, exp_cost 1.591586, grad norm 4.280657, param norm 1423.660522, tps 935.628719, length mean/std 5.000000/0.000000
INFO:root:epoch 41, iter 6000, cost 1.775624, exp_cost 1.593046, grad norm 4.870620, param norm 1423.921143, tps 936.112406, length mean/std 6.000000/0.000000
INFO:root:epoch 41, iter 6200, cost 1.815450, exp_cost 1.600852, grad norm 5.829322, param norm 1424.208496, tps 936.590692, length mean/std 7.000000/0.000000
INFO:root:epoch 41, iter 6400, cost 1.596902, exp_cost 1.602111, grad norm 22.423264, param norm 1424.545288, tps 937.186691, length mean/std 36.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 124366182 get requests, put_count=124366471 evicted_count=134000 eviction_rate=0.00107746 and unsatisfied allocation rate=0.00107866
INFO:root:epoch 41, iter 6600, cost 1.618828, exp_cost 1.603624, grad norm 6.978151, param norm 1424.810181, tps 937.737846, length mean/std 9.000000/0.000000
INFO:root:epoch 41, iter 6800, cost 1.524601, exp_cost 1.594966, grad norm 13.230407, param norm 1425.150146, tps 938.221029, length mean/std 20.000000/0.000000
INFO:root:epoch 41, iter 7000, cost 1.528059, exp_cost 1.606542, grad norm 11.477240, param norm 1425.441406, tps 938.698753, length mean/std 17.000000/0.000000
INFO:root:epoch 41, iter 7200, cost 1.341460, exp_cost 1.605998, grad norm 5.566356, param norm 1425.716919, tps 939.173623, length mean/std 9.000000/0.000000
INFO:root:epoch 41, iter 7400, cost 1.411650, exp_cost 1.598109, grad norm 8.006574, param norm 1425.975098, tps 939.656373, length mean/std 12.000000/0.000000
INFO:root:epoch 41, iter 7600, cost 1.689918, exp_cost 1.598494, grad norm 17.012289, param norm 1426.330200, tps 940.169209, length mean/std 26.000000/0.000000
INFO:root:epoch 41, iter 7800, cost 1.705952, exp_cost 1.601255, grad norm 3.330880, param norm 1426.504150, tps 940.635493, length mean/std 4.000000/0.000000
INFO:root:epoch 41, iter 8000, cost 1.639698, exp_cost 1.602417, grad norm 3.841792, param norm 1426.863892, tps 941.088455, length mean/std 5.000000/0.000000
INFO:root:epoch 41, iter 8200, cost 1.608398, exp_cost 1.603217, grad norm 5.010101, param norm 1427.180420, tps 941.514759, length mean/std 7.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 131199609 get requests, put_count=131199759 evicted_count=144000 eviction_rate=0.00109756 and unsatisfied allocation rate=0.00109976
INFO:root:epoch 41, iter 8400, cost 2.031101, exp_cost 1.590330, grad norm 1.584982, param norm 1427.454590, tps 941.928480, length mean/std 2.000000/0.000000
INFO:root:epoch 41, iter 8600, cost 1.552630, exp_cost 1.597019, grad norm 13.732573, param norm 1427.797363, tps 942.454967, length mean/std 22.000000/0.000000
INFO:root:epoch 41, iter 8800, cost 2.014776, exp_cost 1.607962, grad norm 1.681175, param norm 1428.039551, tps 942.889401, length mean/std 2.000000/0.000000
INFO:root:epoch 41, iter 9000, cost 2.104270, exp_cost 1.606129, grad norm 1.798186, param norm 1428.372192, tps 943.391506, length mean/std 2.000000/0.000000
INFO:root:epoch 41, iter 9200, cost 2.172819, exp_cost 1.598778, grad norm 1.622907, param norm 1428.604614, tps 943.782966, length mean/std 2.000000/0.000000
INFO:root:epoch 41, iter 9400, cost 1.635456, exp_cost 1.603300, grad norm 14.826826, param norm 1428.957031, tps 944.218828, length mean/std 23.000000/0.000000
INFO:root:epoch 41, iter 9600, cost 1.546831, exp_cost 1.599213, grad norm 10.380286, param norm 1429.246826, tps 944.596400, length mean/std 16.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 136990921 get requests, put_count=136991212 evicted_count=154000 eviction_rate=0.00112416 and unsatisfied allocation rate=0.00112524
INFO:root:epoch 41, iter 9800, cost 1.549671, exp_cost 1.597785, grad norm 6.399246, param norm 1429.520630, tps 945.037555, length mean/std 9.000000/0.000000
INFO:root:epoch 41, iter 10000, cost 1.766687, exp_cost 1.602120, grad norm 4.953507, param norm 1429.842651, tps 945.521836, length mean/std 6.000000/0.000000
INFO:root:epoch 41, iter 10200, cost 1.472852, exp_cost 1.602437, grad norm 15.129345, param norm 1430.139648, tps 945.909885, length mean/std 24.000000/0.000000
INFO:root:epoch 41, iter 10400, cost 1.680111, exp_cost 1.600978, grad norm 17.886398, param norm 1430.463989, tps 946.384595, length mean/std 30.000000/0.000000
INFO:root:epoch 41, iter 10600, cost 1.633294, exp_cost 1.616673, grad norm 12.672421, param norm 1430.771606, tps 946.806639, length mean/std 19.000000/0.000000
INFO:root:epoch 41, iter 10800, cost 2.975895, exp_cost 1.600198, grad norm 0.754675, param norm 1431.014404, tps 947.191616, length mean/std 1.000000/0.000000
INFO:root:epoch 41, iter 11000, cost 1.562616, exp_cost 1.597953, grad norm 6.768000, param norm 1431.302979, tps 947.563015, length mean/std 10.000000/0.000000
INFO:root:epoch 41, iter 11200, cost 1.531665, exp_cost 1.603558, grad norm 11.345165, param norm 1431.600342, tps 947.943305, length mean/std 19.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 142622593 get requests, put_count=142622921 evicted_count=164000 eviction_rate=0.00114989 and unsatisfied allocation rate=0.00115066
INFO:root:epoch 41, iter 11400, cost 2.899783, exp_cost 1.614496, grad norm 0.725568, param norm 1431.848755, tps 948.334991, length mean/std 1.000000/0.000000
INFO:root:epoch 41, iter 11600, cost 1.628579, exp_cost 1.607101, grad norm 17.447361, param norm 1432.146240, tps 948.771611, length mean/std 26.000000/0.000000
INFO:root:epoch 41, iter 11800, cost 1.648208, exp_cost 1.604816, grad norm 23.886562, param norm 1432.450684, tps 949.198953, length mean/std 37.000000/0.000000
INFO:root:epoch 41, iter 12000, cost 1.535652, exp_cost 1.603341, grad norm 15.216203, param norm 1432.756348, tps 949.639847, length mean/std 26.000000/0.000000
INFO:root:epoch 41, iter 12200, cost 1.511036, exp_cost 1.602839, grad norm 11.467004, param norm 1433.036621, tps 950.015262, length mean/std 18.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 146962849 get requests, put_count=146963080 evicted_count=174000 eviction_rate=0.00118397 and unsatisfied allocation rate=0.00118538
INFO:root:epoch 41, iter 12400, cost 2.916532, exp_cost 1.613574, grad norm 0.584619, param norm 1433.282104, tps 950.432844, length mean/std 1.000000/0.000000
INFO:root:Epoch 41 Validation cost: 1.700827 time: 6663.344431
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-41 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-41 is not in all_model_checkpoint_paths. Manually adding it.
INFO:root:epoch 42, iter 200, cost 1.589170, exp_cost 1.583355, grad norm 16.791227, param norm 1433.788940, tps 921.250744, length mean/std 26.000000/0.000000
INFO:root:epoch 42, iter 400, cost 1.531600, exp_cost 1.582697, grad norm 18.674892, param norm 1434.086914, tps 921.770712, length mean/std 28.000000/0.000000
INFO:root:epoch 42, iter 600, cost 1.424397, exp_cost 1.586407, grad norm 10.164085, param norm 1434.368652, tps 922.271931, length mean/std 16.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 152241308 get requests, put_count=152241564 evicted_count=184000 eviction_rate=0.00120861 and unsatisfied allocation rate=0.0012098
INFO:root:epoch 42, iter 800, cost 1.705327, exp_cost 1.585795, grad norm 6.251794, param norm 1434.652466, tps 922.862343, length mean/std 8.000000/0.000000
INFO:root:epoch 42, iter 1000, cost 2.181051, exp_cost 1.581765, grad norm 2.003009, param norm 1434.841064, tps 923.330931, length mean/std 2.000000/0.000000
INFO:root:epoch 42, iter 1200, cost 1.581056, exp_cost 1.584419, grad norm 16.750847, param norm 1435.151245, tps 923.822784, length mean/std 24.000000/0.000000
INFO:root:epoch 42, iter 1400, cost 1.556865, exp_cost 1.584481, grad norm 7.080862, param norm 1435.428467, tps 924.331388, length mean/std 10.000000/0.000000
INFO:root:epoch 42, iter 1600, cost 1.538601, exp_cost 1.583863, grad norm 10.025014, param norm 1435.728760, tps 924.789305, length mean/std 15.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 156181966 get requests, put_count=156182158 evicted_count=194000 eviction_rate=0.00124214 and unsatisfied allocation rate=0.00124372
INFO:root:epoch 42, iter 1800, cost 1.752124, exp_cost 1.582035, grad norm 5.367198, param norm 1436.064819, tps 925.240689, length mean/std 7.000000/0.000000
INFO:root:epoch 42, iter 2000, cost 1.557197, exp_cost 1.582875, grad norm 19.529598, param norm 1436.246216, tps 925.746416, length mean/std 33.000000/0.000000
INFO:root:epoch 42, iter 2200, cost 1.526777, exp_cost 1.597777, grad norm 9.806269, param norm 1436.526245, tps 926.148120, length mean/std 14.000000/0.000000
INFO:root:epoch 42, iter 2400, cost 1.490190, exp_cost 1.583123, grad norm 9.854057, param norm 1436.842407, tps 926.588333, length mean/std 15.000000/0.000000
INFO:root:epoch 42, iter 2600, cost 1.574070, exp_cost 1.592080, grad norm 17.827652, param norm 1437.109985, tps 927.054968, length mean/std 27.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 159798749 get requests, put_count=159798684 evicted_count=204000 eviction_rate=0.00127661 and unsatisfied allocation rate=0.00127975
INFO:root:epoch 42, iter 2800, cost 1.482419, exp_cost 1.583314, grad norm 13.723677, param norm 1437.408447, tps 927.494066, length mean/std 23.000000/0.000000
INFO:root:epoch 42, iter 3000, cost 1.607529, exp_cost 1.593432, grad norm 16.396954, param norm 1437.688477, tps 927.947774, length mean/std 25.000000/0.000000
INFO:root:epoch 42, iter 3200, cost 1.834644, exp_cost 1.591524, grad norm 2.553008, param norm 1437.946411, tps 928.420186, length mean/std 3.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 162537119 get requests, put_count=162537445 evicted_count=214000 eviction_rate=0.00131662 and unsatisfied allocation rate=0.00131731
INFO:root:epoch 42, iter 3400, cost 1.606820, exp_cost 1.588471, grad norm 5.647143, param norm 1438.221436, tps 928.875695, length mean/std 8.000000/0.000000
INFO:root:epoch 42, iter 3600, cost 1.743611, exp_cost 1.596847, grad norm 21.160191, param norm 1438.528809, tps 929.362552, length mean/std 33.000000/0.000000
INFO:root:epoch 42, iter 3800, cost 1.552198, exp_cost 1.596922, grad norm 12.763120, param norm 1438.844727, tps 929.792160, length mean/std 18.000000/0.000000
INFO:root:epoch 42, iter 4000, cost 1.679971, exp_cost 1.586234, grad norm 5.671640, param norm 1439.139404, tps 930.181413, length mean/std 7.000000/0.000000
INFO:root:epoch 42, iter 4200, cost 1.791674, exp_cost 1.589115, grad norm 5.553370, param norm 1439.450928, tps 930.611528, length mean/std 7.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 166020045 get requests, put_count=166020250 evicted_count=224000 eviction_rate=0.00134923 and unsatisfied allocation rate=0.00135064
INFO:root:epoch 42, iter 4400, cost 1.640541, exp_cost 1.587816, grad norm 14.808756, param norm 1439.754761, tps 931.034169, length mean/std 22.000000/0.000000
INFO:root:epoch 42, iter 4600, cost 1.598129, exp_cost 1.578140, grad norm 8.602456, param norm 1440.032837, tps 931.463630, length mean/std 12.000000/0.000000
INFO:root:epoch 42, iter 4800, cost 1.681303, exp_cost 1.597590, grad norm 10.220978, param norm 1440.350952, tps 931.906554, length mean/std 14.000000/0.000000
INFO:root:epoch 42, iter 5000, cost 1.626476, exp_cost 1.591507, grad norm 21.562372, param norm 1440.592896, tps 932.313755, length mean/std 32.000000/0.000000
INFO:root:epoch 42, iter 5200, cost 1.538244, exp_cost 1.583029, grad norm 18.737274, param norm 1440.856323, tps 932.736814, length mean/std 32.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 170417176 get requests, put_count=170417527 evicted_count=234000 eviction_rate=0.0013731 and unsatisfied allocation rate=0.00137361
INFO:root:epoch 42, iter 5400, cost 1.645206, exp_cost 1.591019, grad norm 22.034578, param norm 1441.135620, tps 933.154722, length mean/std 37.000000/0.000000
INFO:root:epoch 42, iter 5600, cost 1.571136, exp_cost 1.596666, grad norm 13.609385, param norm 1441.495728, tps 933.546535, length mean/std 20.000000/0.000000
INFO:root:epoch 42, iter 5800, cost 1.794272, exp_cost 1.591494, grad norm 5.232675, param norm 1441.768799, tps 933.954484, length mean/std 6.000000/0.000000
INFO:root:epoch 42, iter 6000, cost 1.481742, exp_cost 1.587925, grad norm 18.255219, param norm 1442.017944, tps 934.352750, length mean/std 31.000000/0.000000
INFO:root:epoch 42, iter 6200, cost 1.475224, exp_cost 1.613111, grad norm 7.508995, param norm 1442.357178, tps 934.747042, length mean/std 12.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 174220966 get requests, put_count=174220859 evicted_count=244000 eviction_rate=0.00140052 and unsatisfied allocation rate=0.00140365
INFO:root:epoch 42, iter 6400, cost 1.850205, exp_cost 1.599677, grad norm 3.915304, param norm 1442.577637, tps 935.069082, length mean/std 5.000000/0.000000
INFO:root:epoch 42, iter 6600, cost 1.301411, exp_cost 1.600207, grad norm 5.729342, param norm 1442.886475, tps 935.403860, length mean/std 9.000000/0.000000
INFO:root:epoch 42, iter 6800, cost 1.685006, exp_cost 1.604438, grad norm 25.307661, param norm 1443.120483, tps 935.777729, length mean/std 42.000000/0.000000
INFO:root:epoch 42, iter 7000, cost 1.578900, exp_cost 1.607479, grad norm 7.927894, param norm 1443.416260, tps 936.161653, length mean/std 11.000000/0.000000
INFO:root:epoch 42, iter 7200, cost 1.414333, exp_cost 1.589868, grad norm 9.064009, param norm 1443.686035, tps 936.527161, length mean/std 15.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 177949488 get requests, put_count=177949695 evicted_count=254000 eviction_rate=0.00142737 and unsatisfied allocation rate=0.00142867
INFO:root:epoch 42, iter 7400, cost 1.486591, exp_cost 1.593815, grad norm 9.941689, param norm 1444.018555, tps 936.893782, length mean/std 15.000000/0.000000
INFO:root:epoch 42, iter 7600, cost 1.689469, exp_cost 1.596898, grad norm 7.530821, param norm 1444.239624, tps 937.258558, length mean/std 10.000000/0.000000
INFO:root:epoch 42, iter 7800, cost 1.566636, exp_cost 1.595901, grad norm 14.056231, param norm 1444.670898, tps 937.663181, length mean/std 23.000000/0.000000
INFO:root:epoch 42, iter 8000, cost 1.369648, exp_cost 1.592877, grad norm 13.680258, param norm 1444.983032, tps 938.009223, length mean/std 25.000000/0.000000
INFO:root:epoch 42, iter 8200, cost 2.916772, exp_cost 1.610003, grad norm 0.799719, param norm 1445.272949, tps 938.388157, length mean/std 1.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 181820775 get requests, put_count=181820730 evicted_count=264000 eviction_rate=0.00145198 and unsatisfied allocation rate=0.00145464
INFO:root:epoch 42, iter 8400, cost 1.588482, exp_cost 1.606579, grad norm 15.598927, param norm 1445.576050, tps 938.795557, length mean/std 24.000000/0.000000
INFO:root:epoch 42, iter 8600, cost 1.782607, exp_cost 1.598842, grad norm 3.994228, param norm 1445.781982, tps 939.147835, length mean/std 5.000000/0.000000
INFO:root:epoch 42, iter 8800, cost 1.436691, exp_cost 1.587172, grad norm 10.268049, param norm 1446.113403, tps 939.510481, length mean/std 17.000000/0.000000
INFO:root:epoch 42, iter 9000, cost 1.555644, exp_cost 1.602368, grad norm 15.514305, param norm 1446.339722, tps 939.850071, length mean/std 26.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 185497928 get requests, put_count=185498282 evicted_count=274000 eviction_rate=0.0014771 and unsatisfied allocation rate=0.00147756
INFO:root:epoch 42, iter 9200, cost 1.791265, exp_cost 1.608341, grad norm 3.297410, param norm 1446.679932, tps 940.188745, length mean/std 4.000000/0.000000
INFO:root:epoch 42, iter 9400, cost 1.484374, exp_cost 1.611923, grad norm 5.239326, param norm 1446.943359, tps 940.533778, length mean/std 8.000000/0.000000
INFO:root:epoch 42, iter 9600, cost 1.549488, exp_cost 1.599616, grad norm 17.795630, param norm 1447.229492, tps 940.868178, length mean/std 29.000000/0.000000
INFO:root:epoch 42, iter 9800, cost 1.609608, exp_cost 1.597018, grad norm 18.763302, param norm 1447.591309, tps 941.225038, length mean/std 30.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 188791264 get requests, put_count=188791617 evicted_count=284000 eviction_rate=0.0015043 and unsatisfied allocation rate=0.00150476
INFO:root:epoch 42, iter 10000, cost 1.631210, exp_cost 1.605969, grad norm 13.891466, param norm 1447.856201, tps 941.584523, length mean/std 20.000000/0.000000
INFO:root:epoch 42, iter 10200, cost 1.619573, exp_cost 1.602046, grad norm 20.180504, param norm 1448.116089, tps 941.873787, length mean/std 33.000000/0.000000
INFO:root:epoch 42, iter 10400, cost 1.664113, exp_cost 1.593887, grad norm 18.888229, param norm 1448.402710, tps 942.227310, length mean/std 28.000000/0.000000
INFO:root:epoch 42, iter 10600, cost 1.505928, exp_cost 1.593122, grad norm 17.081516, param norm 1448.740723, tps 942.547072, length mean/std 27.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 191916878 get requests, put_count=191917206 evicted_count=294000 eviction_rate=0.00153191 and unsatisfied allocation rate=0.00153249
INFO:root:epoch 42, iter 10800, cost 1.511638, exp_cost 1.593254, grad norm 7.412844, param norm 1449.034180, tps 942.884862, length mean/std 11.000000/0.000000
INFO:root:epoch 42, iter 11000, cost 1.664392, exp_cost 1.602044, grad norm 17.157187, param norm 1449.258911, tps 943.255786, length mean/std 25.000000/0.000000
INFO:root:epoch 42, iter 11200, cost 1.522571, exp_cost 1.600364, grad norm 11.732214, param norm 1449.507935, tps 943.598008, length mean/std 18.000000/0.000000
INFO:root:epoch 42, iter 11400, cost 1.611511, exp_cost 1.600822, grad norm 9.889287, param norm 1449.755981, tps 943.900470, length mean/std 15.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 195035673 get requests, put_count=195035736 evicted_count=304000 eviction_rate=0.00155869 and unsatisfied allocation rate=0.00156061
INFO:root:epoch 42, iter 11600, cost 1.547795, exp_cost 1.607628, grad norm 7.068254, param norm 1450.081055, tps 944.239279, length mean/std 10.000000/0.000000
INFO:root:epoch 42, iter 11800, cost 1.659105, exp_cost 1.612687, grad norm 21.719990, param norm 1450.369751, tps 944.536309, length mean/std 35.000000/0.000000
INFO:root:epoch 42, iter 12000, cost 2.222346, exp_cost 1.603110, grad norm 2.807339, param norm 1450.558228, tps 944.855398, length mean/std 3.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 197509388 get requests, put_count=197509654 evicted_count=314000 eviction_rate=0.0015898 and unsatisfied allocation rate=0.00159067
INFO:root:epoch 42, iter 12200, cost 2.266434, exp_cost 1.597692, grad norm 1.848607, param norm 1450.875488, tps 945.163393, length mean/std 2.000000/0.000000
INFO:root:epoch 42, iter 12400, cost 1.540870, exp_cost 1.598723, grad norm 16.543350, param norm 1451.150269, tps 945.478960, length mean/std 27.000000/0.000000
INFO:root:Epoch 42 Validation cost: 1.701320 time: 6757.150332
INFO:root:Annealing learning rate by 0.950000
INFO:root:epoch 43, iter 200, cost 1.796541, exp_cost 1.586971, grad norm 4.074457, param norm 1433.779541, tps 923.846610, length mean/std 5.000000/0.000000
INFO:root:epoch 43, iter 400, cost 1.477227, exp_cost 1.584067, grad norm 13.119660, param norm 1434.139404, tps 924.261987, length mean/std 21.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 202268991 get requests, put_count=202268967 evicted_count=324000 eviction_rate=0.00160183 and unsatisfied allocation rate=0.00160411
INFO:root:epoch 43, iter 600, cost 1.967646, exp_cost 1.582761, grad norm 1.748270, param norm 1434.353760, tps 924.657583, length mean/std 2.000000/0.000000
INFO:root:epoch 43, iter 800, cost 1.709881, exp_cost 1.597470, grad norm 20.827528, param norm 1434.696655, tps 924.980136, length mean/std 32.000000/0.000000
INFO:root:epoch 43, iter 1000, cost 1.478435, exp_cost 1.601490, grad norm 14.838512, param norm 1435.001953, tps 925.351991, length mean/std 24.000000/0.000000
INFO:root:epoch 43, iter 1200, cost 1.489028, exp_cost 1.592162, grad norm 7.115830, param norm 1435.201660, tps 925.681802, length mean/std 10.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 205284446 get requests, put_count=205284591 evicted_count=334000 eviction_rate=0.00162701 and unsatisfied allocation rate=0.00162844
INFO:root:epoch 43, iter 1400, cost 1.544105, exp_cost 1.580674, grad norm 16.137348, param norm 1435.590698, tps 926.094071, length mean/std 26.000000/0.000000
INFO:root:epoch 43, iter 1600, cost 1.588648, exp_cost 1.582787, grad norm 11.672232, param norm 1435.810913, tps 926.472010, length mean/std 19.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 207513354 get requests, put_count=207513571 evicted_count=344000 eviction_rate=0.00165772 and unsatisfied allocation rate=0.00165879
INFO:root:epoch 43, iter 1800, cost 1.990473, exp_cost 1.585690, grad norm 2.903555, param norm 1436.052490, tps 926.828893, length mean/std 3.000000/0.000000
INFO:root:epoch 43, iter 2000, cost 1.605280, exp_cost 1.595654, grad norm 19.497477, param norm 1436.343018, tps 927.202221, length mean/std 32.000000/0.000000
INFO:root:epoch 43, iter 2200, cost 1.990983, exp_cost 1.591925, grad norm 2.677489, param norm 1436.652588, tps 927.549080, length mean/std 3.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 209589843 get requests, put_count=209589766 evicted_count=354000 eviction_rate=0.00168901 and unsatisfied allocation rate=0.00169147
INFO:root:epoch 43, iter 2400, cost 1.618561, exp_cost 1.598719, grad norm 21.522373, param norm 1436.978516, tps 927.937392, length mean/std 35.000000/0.000000
INFO:root:epoch 43, iter 2600, cost 2.164383, exp_cost 1.593767, grad norm 1.749472, param norm 1437.243896, tps 928.271959, length mean/std 2.000000/0.000000
INFO:root:epoch 43, iter 2800, cost 1.510047, exp_cost 1.598216, grad norm 11.958280, param norm 1437.493896, tps 928.641581, length mean/std 19.000000/0.000000
INFO:root:epoch 43, iter 3000, cost 1.717637, exp_cost 1.593055, grad norm 5.267466, param norm 1437.753418, tps 929.002667, length mean/std 6.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 212484399 get requests, put_count=212484662 evicted_count=364000 eviction_rate=0.00171306 and unsatisfied allocation rate=0.00171389
INFO:root:epoch 43, iter 3200, cost 1.516857, exp_cost 1.598109, grad norm 19.875692, param norm 1438.170654, tps 929.363120, length mean/std 30.000000/0.000000
INFO:root:epoch 43, iter 3400, cost 2.032611, exp_cost 1.594468, grad norm 2.708638, param norm 1438.444946, tps 929.674552, length mean/std 3.000000/0.000000
INFO:root:epoch 43, iter 3600, cost 1.614094, exp_cost 1.585263, grad norm 6.572235, param norm 1438.673706, tps 929.993696, length mean/std 9.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 215142825 get requests, put_count=215142889 evicted_count=374000 eviction_rate=0.00173838 and unsatisfied allocation rate=0.00174012
INFO:root:epoch 43, iter 3800, cost 1.683717, exp_cost 1.594649, grad norm 6.130717, param norm 1438.902466, tps 930.349603, length mean/std 8.000000/0.000000
INFO:root:epoch 43, iter 4000, cost 1.907722, exp_cost 1.594314, grad norm 3.586086, param norm 1439.235840, tps 930.689313, length mean/std 4.000000/0.000000
INFO:root:epoch 43, iter 4200, cost 1.623298, exp_cost 1.599796, grad norm 8.701447, param norm 1439.552490, tps 931.030550, length mean/std 11.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 217547061 get requests, put_count=217547142 evicted_count=384000 eviction_rate=0.00176513 and unsatisfied allocation rate=0.00176678
INFO:root:epoch 43, iter 4400, cost 1.571063, exp_cost 1.613264, grad norm 18.839108, param norm 1439.861816, tps 931.352369, length mean/std 32.000000/0.000000
INFO:root:epoch 43, iter 4600, cost 1.555964, exp_cost 1.591896, grad norm 6.143304, param norm 1440.173584, tps 931.687492, length mean/std 9.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 219536264 get requests, put_count=219536592 evicted_count=394000 eviction_rate=0.00179469 and unsatisfied allocation rate=0.00179519
INFO:root:epoch 43, iter 4800, cost 1.555374, exp_cost 1.590337, grad norm 18.816477, param norm 1440.450562, tps 931.993985, length mean/std 31.000000/0.000000
INFO:root:epoch 43, iter 5000, cost 1.571891, exp_cost 1.583176, grad norm 15.125787, param norm 1440.757690, tps 932.318341, length mean/std 25.000000/0.000000
INFO:root:epoch 43, iter 5200, cost 1.635999, exp_cost 1.605338, grad norm 17.108913, param norm 1441.025879, tps 932.655636, length mean/std 26.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 221767885 get requests, put_count=221768228 evicted_count=404000 eviction_rate=0.00182172 and unsatisfied allocation rate=0.00182215
INFO:root:epoch 43, iter 5400, cost 1.619936, exp_cost 1.595807, grad norm 20.899755, param norm 1441.192505, tps 932.959884, length mean/std 33.000000/0.000000
INFO:root:epoch 43, iter 5600, cost 1.586787, exp_cost 1.597674, grad norm 2.831475, param norm 1441.504517, tps 933.255907, length mean/std 4.000000/0.000000
INFO:root:epoch 43, iter 5800, cost 1.601354, exp_cost 1.602409, grad norm 6.014140, param norm 1441.821533, tps 933.597990, length mean/std 8.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 224083377 get requests, put_count=224083699 evicted_count=414000 eviction_rate=0.00184752 and unsatisfied allocation rate=0.00184804
INFO:root:epoch 43, iter 6000, cost 1.530215, exp_cost 1.603371, grad norm 4.237672, param norm 1442.140991, tps 933.909879, length mean/std 6.000000/0.000000
INFO:root:epoch 43, iter 6200, cost 1.575940, exp_cost 1.594173, grad norm 18.463779, param norm 1442.433350, tps 934.203460, length mean/std 26.000000/0.000000
INFO:root:epoch 43, iter 6400, cost 1.530920, exp_cost 1.593952, grad norm 13.316694, param norm 1442.741089, tps 934.519912, length mean/std 21.000000/0.000000
INFO:root:epoch 43, iter 6600, cost 1.448361, exp_cost 1.596461, grad norm 6.453617, param norm 1443.063721, tps 934.863955, length mean/std 10.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 227215310 get requests, put_count=227215652 evicted_count=424000 eviction_rate=0.00186607 and unsatisfied allocation rate=0.00186649
INFO:root:epoch 43, iter 6800, cost 2.079129, exp_cost 1.603520, grad norm 1.907788, param norm 1443.314697, tps 935.178219, length mean/std 2.000000/0.000000
INFO:root:epoch 43, iter 7000, cost 1.800604, exp_cost 1.606622, grad norm 5.059773, param norm 1443.575928, tps 935.467229, length mean/std 6.000000/0.000000
INFO:root:epoch 43, iter 7200, cost 1.475012, exp_cost 1.600803, grad norm 6.175629, param norm 1443.952637, tps 935.806147, length mean/std 9.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 229984225 get requests, put_count=229984475 evicted_count=434000 eviction_rate=0.00188708 and unsatisfied allocation rate=0.0018879
INFO:root:epoch 43, iter 7400, cost 1.627638, exp_cost 1.601680, grad norm 23.322002, param norm 1444.184937, tps 936.161075, length mean/std 41.000000/0.000000
INFO:root:epoch 43, iter 7600, cost 1.667385, exp_cost 1.597696, grad norm 19.635235, param norm 1444.469238, tps 936.454245, length mean/std 30.000000/0.000000
INFO:root:epoch 43, iter 7800, cost 1.549015, exp_cost 1.589289, grad norm 15.266803, param norm 1444.732544, tps 936.753381, length mean/std 22.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 232133893 get requests, put_count=232134120 evicted_count=444000 eviction_rate=0.00191269 and unsatisfied allocation rate=0.0019136
INFO:root:epoch 43, iter 8000, cost 1.536363, exp_cost 1.603536, grad norm 14.223122, param norm 1444.922485, tps 937.050991, length mean/std 22.000000/0.000000
INFO:root:epoch 43, iter 8200, cost 1.593157, exp_cost 1.616381, grad norm 11.302035, param norm 1445.247559, tps 937.332653, length mean/std 17.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 234008256 get requests, put_count=234008347 evicted_count=454000 eviction_rate=0.0019401 and unsatisfied allocation rate=0.00194159
INFO:root:epoch 43, iter 8400, cost 1.593834, exp_cost 1.599061, grad norm 14.653316, param norm 1445.531494, tps 937.630864, length mean/std 23.000000/0.000000
INFO:root:epoch 43, iter 8600, cost 1.672414, exp_cost 1.602409, grad norm 3.841292, param norm 1445.798828, tps 937.974383, length mean/std 5.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 235810285 get requests, put_count=235810293 evicted_count=464000 eviction_rate=0.00196768 and unsatisfied allocation rate=0.00196951
INFO:root:epoch 43, iter 8800, cost 2.151734, exp_cost 1.600674, grad norm 2.870921, param norm 1446.104736, tps 938.262182, length mean/std 3.000000/0.000000
INFO:root:epoch 43, iter 9000, cost 1.678797, exp_cost 1.596548, grad norm 8.962942, param norm 1446.325439, tps 938.542957, length mean/std 12.000000/0.000000
INFO:root:epoch 43, iter 9200, cost 1.933633, exp_cost 1.604767, grad norm 4.422831, param norm 1446.586670, tps 938.810104, length mean/std 5.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 238292918 get requests, put_count=238293014 evicted_count=474000 eviction_rate=0.00198915 and unsatisfied allocation rate=0.00199058
INFO:root:epoch 43, iter 9400, cost 2.054177, exp_cost 1.608677, grad norm 4.576096, param norm 1446.859619, tps 939.065588, length mean/std 5.000000/0.000000
INFO:root:epoch 43, iter 9600, cost 1.600396, exp_cost 1.602717, grad norm 12.359090, param norm 1447.078003, tps 939.324298, length mean/std 19.000000/0.000000
INFO:root:epoch 43, iter 9800, cost 1.540355, exp_cost 1.605680, grad norm 11.886510, param norm 1447.393799, tps 939.603949, length mean/std 17.000000/0.000000
INFO:root:epoch 43, iter 10000, cost 1.843377, exp_cost 1.589755, grad norm 2.560998, param norm 1447.669189, tps 939.875462, length mean/std 3.000000/0.000000
INFO:root:epoch 43, iter 10200, cost 1.604759, exp_cost 1.595473, grad norm 15.311775, param norm 1447.975586, tps 940.196503, length mean/std 24.000000/0.000000
INFO:root:epoch 43, iter 10400, cost 1.711819, exp_cost 1.598926, grad norm 4.706197, param norm 1448.321411, tps 940.443472, length mean/std 6.000000/0.000000
INFO:root:epoch 43, iter 10600, cost 1.669143, exp_cost 1.604292, grad norm 12.984839, param norm 1448.693115, tps 940.711912, length mean/std 19.000000/0.000000
INFO:root:epoch 43, iter 10800, cost 1.384987, exp_cost 1.601619, grad norm 6.276301, param norm 1448.979736, tps 940.995869, length mean/std 9.000000/0.000000
INFO:root:epoch 43, iter 11000, cost 1.569333, exp_cost 1.594710, grad norm 9.437076, param norm 1449.263184, tps 941.271569, length mean/std 14.000000/0.000000
INFO:root:epoch 43, iter 11200, cost 1.588485, exp_cost 1.607240, grad norm 19.250652, param norm 1449.537476, tps 941.544732, length mean/std 33.000000/0.000000
INFO:root:epoch 43, iter 11400, cost 1.559309, exp_cost 1.595616, grad norm 7.615213, param norm 1449.791504, tps 941.824081, length mean/std 10.000000/0.000000
INFO:root:epoch 43, iter 11600, cost 1.620645, exp_cost 1.585960, grad norm 10.145256, param norm 1450.063721, tps 942.045171, length mean/std 15.000000/0.000000
INFO:root:epoch 43, iter 11800, cost 1.583143, exp_cost 1.592260, grad norm 19.993574, param norm 1450.410522, tps 942.306761, length mean/std 32.000000/0.000000
INFO:root:epoch 43, iter 12000, cost 1.628928, exp_cost 1.596469, grad norm 21.478352, param norm 1450.718506, tps 942.582072, length mean/std 35.000000/0.000000
INFO:root:epoch 43, iter 12200, cost 1.809205, exp_cost 1.593926, grad norm 5.508601, param norm 1451.048950, tps 942.834150, length mean/std 6.000000/0.000000
INFO:root:epoch 43, iter 12400, cost 1.562523, exp_cost 1.602603, grad norm 10.015076, param norm 1451.269897, tps 943.090007, length mean/std 15.000000/0.000000
INFO:root:Epoch 43 Validation cost: 1.702683 time: 6858.204052
INFO:root:Annealing learning rate by 0.950000
INFO:root:epoch 44, iter 200, cost 1.517089, exp_cost 1.590201, grad norm 12.083992, param norm 1433.744263, tps 925.662613, length mean/std 20.000000/0.000000
INFO:root:epoch 44, iter 400, cost 1.503894, exp_cost 1.583628, grad norm 13.590150, param norm 1433.987671, tps 926.003107, length mean/std 21.000000/0.000000
INFO:root:epoch 44, iter 600, cost 1.577020, exp_cost 1.591252, grad norm 11.135225, param norm 1434.321411, tps 926.324444, length mean/std 16.000000/0.000000
INFO:root:epoch 44, iter 800, cost 1.557780, exp_cost 1.588459, grad norm 9.881961, param norm 1434.584961, tps 926.634479, length mean/std 16.000000/0.000000
INFO:root:epoch 44, iter 1000, cost 1.480412, exp_cost 1.583619, grad norm 5.575618, param norm 1434.928101, tps 926.916317, length mean/std 8.000000/0.000000
INFO:root:epoch 44, iter 1200, cost 1.519680, exp_cost 1.595006, grad norm 14.712395, param norm 1435.181274, tps 927.187169, length mean/std 23.000000/0.000000
INFO:root:epoch 44, iter 1400, cost 1.559122, exp_cost 1.585902, grad norm 9.876255, param norm 1435.383911, tps 927.473866, length mean/std 15.000000/0.000000
INFO:root:epoch 44, iter 1600, cost 1.508119, exp_cost 1.582093, grad norm 7.694047, param norm 1435.705078, tps 927.789320, length mean/std 12.000000/0.000000
INFO:root:epoch 44, iter 1800, cost 1.512945, exp_cost 1.569725, grad norm 7.856736, param norm 1436.000366, tps 928.118917, length mean/std 11.000000/0.000000
INFO:root:epoch 44, iter 2000, cost 1.480371, exp_cost 1.587112, grad norm 8.883576, param norm 1436.295288, tps 928.461044, length mean/std 13.000000/0.000000
INFO:root:epoch 44, iter 2200, cost 1.581324, exp_cost 1.590166, grad norm 26.728121, param norm 1436.475342, tps 928.704928, length mean/std 43.000000/0.000000
INFO:root:epoch 44, iter 2400, cost 1.576359, exp_cost 1.590542, grad norm 10.526642, param norm 1436.784180, tps 929.020301, length mean/std 16.000000/0.000000
INFO:root:epoch 44, iter 2600, cost 1.530081, exp_cost 1.589190, grad norm 9.367349, param norm 1437.065918, tps 929.295203, length mean/std 14.000000/0.000000
INFO:root:epoch 44, iter 2800, cost 1.470309, exp_cost 1.598092, grad norm 16.327717, param norm 1437.350830, tps 929.583382, length mean/std 23.000000/0.000000
INFO:root:epoch 44, iter 3000, cost 1.490464, exp_cost 1.588517, grad norm 10.576002, param norm 1437.631836, tps 929.880062, length mean/std 17.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 24813108 get requests, put_count=24813037 evicted_count=3000 eviction_rate=0.000120904 and unsatisfied allocation rate=0.00014319
INFO:root:epoch 44, iter 3200, cost 1.462824, exp_cost 1.592950, grad norm 11.393823, param norm 1437.900513, tps 930.195957, length mean/std 19.000000/0.000000
INFO:root:epoch 44, iter 3400, cost 1.655619, exp_cost 1.599432, grad norm 22.010767, param norm 1438.206787, tps 930.460263, length mean/std 33.562500/0.496078
INFO:root:epoch 44, iter 3600, cost 1.695342, exp_cost 1.586752, grad norm 4.422362, param norm 1438.430176, tps 930.730414, length mean/std 6.000000/0.000000
INFO:root:epoch 44, iter 3800, cost 1.534107, exp_cost 1.584575, grad norm 28.130192, param norm 1438.675415, tps 931.023731, length mean/std 42.000000/0.000000
INFO:root:epoch 44, iter 4000, cost 2.134639, exp_cost 1.592404, grad norm 3.486326, param norm 1439.027954, tps 931.320707, length mean/std 4.000000/0.000000
INFO:root:epoch 44, iter 4200, cost 1.450018, exp_cost 1.603036, grad norm 10.035195, param norm 1439.363892, tps 931.617272, length mean/std 16.000000/0.000000
INFO:root:epoch 44, iter 4400, cost 1.587842, exp_cost 1.605044, grad norm 3.326603, param norm 1439.598755, tps 931.888273, length mean/std 4.000000/0.000000
INFO:root:epoch 44, iter 4600, cost 1.462018, exp_cost 1.592165, grad norm 9.542044, param norm 1439.867432, tps 932.183901, length mean/std 14.000000/0.000000
INFO:root:epoch 44, iter 4800, cost 1.679341, exp_cost 1.587066, grad norm 3.322381, param norm 1440.105469, tps 932.426258, length mean/std 4.000000/0.000000
INFO:root:epoch 44, iter 5000, cost 1.631909, exp_cost 1.600212, grad norm 14.354697, param norm 1440.440918, tps 932.706133, length mean/std 21.000000/0.000000
INFO:root:epoch 44, iter 5200, cost 1.673218, exp_cost 1.608415, grad norm 14.224430, param norm 1440.704956, tps 932.996366, length mean/std 21.000000/0.000000
INFO:root:epoch 44, iter 5400, cost 1.668298, exp_cost 1.595065, grad norm 25.457209, param norm 1441.008057, tps 933.278939, length mean/std 40.000000/0.000000
INFO:root:epoch 44, iter 5600, cost 1.546983, exp_cost 1.600103, grad norm 6.435565, param norm 1441.323242, tps 933.556563, length mean/std 9.000000/0.000000
INFO:root:epoch 44, iter 5800, cost 1.425364, exp_cost 1.613105, grad norm 6.040261, param norm 1441.543091, tps 933.771995, length mean/std 9.000000/0.000000
INFO:root:epoch 44, iter 6000, cost 1.670220, exp_cost 1.598921, grad norm 8.738156, param norm 1441.827148, tps 934.028846, length mean/std 12.000000/0.000000
INFO:root:epoch 44, iter 6200, cost 1.611622, exp_cost 1.594162, grad norm 5.062994, param norm 1442.144775, tps 934.299384, length mean/std 7.000000/0.000000
INFO:root:epoch 44, iter 6400, cost 1.589104, exp_cost 1.599309, grad norm 10.674507, param norm 1442.430176, tps 934.597867, length mean/std 16.000000/0.000000
INFO:root:epoch 44, iter 6600, cost 1.434665, exp_cost 1.599210, grad norm 8.720576, param norm 1442.669678, tps 934.845483, length mean/std 13.000000/0.000000
INFO:root:epoch 44, iter 6800, cost 1.591820, exp_cost 1.590718, grad norm 12.745697, param norm 1442.956787, tps 935.119368, length mean/std 20.000000/0.000000
INFO:root:epoch 44, iter 7000, cost 1.479533, exp_cost 1.584061, grad norm 11.012688, param norm 1443.291138, tps 935.354671, length mean/std 17.000000/0.000000
INFO:root:epoch 44, iter 7200, cost 1.824263, exp_cost 1.598505, grad norm 4.829983, param norm 1443.543701, tps 935.576699, length mean/std 6.000000/0.000000
INFO:root:epoch 44, iter 7400, cost 1.450134, exp_cost 1.605897, grad norm 12.996966, param norm 1443.859375, tps 935.833894, length mean/std 21.000000/0.000000
INFO:root:epoch 44, iter 7600, cost 1.732068, exp_cost 1.600927, grad norm 23.636042, param norm 1444.064087, tps 936.070046, length mean/std 37.000000/0.000000
INFO:root:epoch 44, iter 7800, cost 1.575414, exp_cost 1.597486, grad norm 13.031711, param norm 1444.433228, tps 936.336592, length mean/std 21.000000/0.000000
INFO:root:epoch 44, iter 8000, cost 1.591839, exp_cost 1.605531, grad norm 16.211016, param norm 1444.651123, tps 936.559321, length mean/std 26.000000/0.000000
INFO:root:epoch 44, iter 8200, cost 1.565112, exp_cost 1.602888, grad norm 10.010046, param norm 1444.931763, tps 936.801140, length mean/std 16.000000/0.000000
INFO:root:epoch 44, iter 8400, cost 1.594448, exp_cost 1.596910, grad norm 6.508081, param norm 1445.244629, tps 937.036428, length mean/std 9.000000/0.000000
INFO:root:epoch 44, iter 8600, cost 1.576151, exp_cost 1.606966, grad norm 15.852995, param norm 1445.590698, tps 937.300364, length mean/std 25.000000/0.000000
INFO:root:epoch 44, iter 8800, cost 1.607339, exp_cost 1.607540, grad norm 20.129242, param norm 1445.901245, tps 937.561011, length mean/std 33.000000/0.000000
INFO:root:epoch 44, iter 9000, cost 1.612988, exp_cost 1.592293, grad norm 15.737571, param norm 1446.153687, tps 937.798435, length mean/std 24.000000/0.000000
INFO:root:epoch 44, iter 9200, cost 1.655276, exp_cost 1.596549, grad norm 7.233747, param norm 1446.455811, tps 938.047820, length mean/std 10.000000/0.000000
INFO:root:epoch 44, iter 9400, cost 1.541423, exp_cost 1.599117, grad norm 9.499305, param norm 1446.765015, tps 938.295561, length mean/std 14.000000/0.000000
INFO:root:epoch 44, iter 9600, cost 1.467438, exp_cost 1.593185, grad norm 17.933516, param norm 1447.023926, tps 938.536212, length mean/std 33.000000/0.000000
INFO:root:epoch 44, iter 9800, cost 1.573514, exp_cost 1.601274, grad norm 14.197779, param norm 1447.332886, tps 938.752913, length mean/std 23.000000/0.000000
INFO:root:epoch 44, iter 10000, cost 1.575636, exp_cost 1.607180, grad norm 13.348717, param norm 1447.594849, tps 938.995831, length mean/std 22.000000/0.000000
INFO:root:epoch 44, iter 10200, cost 1.482830, exp_cost 1.598234, grad norm 5.584314, param norm 1447.932495, tps 939.267041, length mean/std 8.000000/0.000000
INFO:root:epoch 44, iter 10400, cost 1.546842, exp_cost 1.604966, grad norm 14.218216, param norm 1448.194092, tps 939.484086, length mean/std 22.000000/0.000000
INFO:root:epoch 44, iter 10600, cost 1.865180, exp_cost 1.609166, grad norm 2.720206, param norm 1448.493652, tps 939.725758, length mean/std 3.000000/0.000000
INFO:root:epoch 44, iter 10800, cost 1.528718, exp_cost 1.597482, grad norm 8.304248, param norm 1448.844849, tps 939.935530, length mean/std 12.000000/0.000000
INFO:root:epoch 44, iter 11000, cost 1.669305, exp_cost 1.601235, grad norm 21.445911, param norm 1449.082153, tps 940.135543, length mean/std 35.000000/0.000000
INFO:root:epoch 44, iter 11200, cost 1.732523, exp_cost 1.620334, grad norm 24.980648, param norm 1449.392090, tps 940.382087, length mean/std 44.000000/0.000000
INFO:root:epoch 44, iter 11400, cost 1.860315, exp_cost 1.604576, grad norm 3.712380, param norm 1449.662842, tps 940.622034, length mean/std 4.000000/0.000000
INFO:root:epoch 44, iter 11600, cost 1.623409, exp_cost 1.602040, grad norm 10.349844, param norm 1449.968140, tps 940.825914, length mean/std 15.000000/0.000000
INFO:root:epoch 44, iter 11800, cost 1.581801, exp_cost 1.608319, grad norm 19.973715, param norm 1450.273560, tps 941.050487, length mean/std 34.000000/0.000000
INFO:root:epoch 44, iter 12000, cost 1.561328, exp_cost 1.590813, grad norm 12.916813, param norm 1450.541870, tps 941.248631, length mean/std 22.000000/0.000000
INFO:root:epoch 44, iter 12200, cost 1.610974, exp_cost 1.614037, grad norm 14.408983, param norm 1450.806885, tps 941.455754, length mean/std 24.375000/0.484123
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 62924988 get requests, put_count=62924870 evicted_count=13000 eviction_rate=0.000206596 and unsatisfied allocation rate=0.00021613
INFO:root:epoch 44, iter 12400, cost 1.718782, exp_cost 1.608889, grad norm 24.653738, param norm 1451.075562, tps 941.700563, length mean/std 43.000000/0.000000
INFO:root:Epoch 44 Validation cost: 1.700772 time: 6938.006855
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-44 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-44 is not in all_model_checkpoint_paths. Manually adding it.
INFO:root:epoch 45, iter 200, cost 2.440835, exp_cost 1.588778, grad norm 1.853794, param norm 1451.485596, tps 926.976070, length mean/std 2.000000/0.000000
INFO:root:epoch 45, iter 400, cost 1.578136, exp_cost 1.589421, grad norm 21.949337, param norm 1451.691162, tps 927.243726, length mean/std 39.000000/0.000000
INFO:root:epoch 45, iter 600, cost 1.921749, exp_cost 1.579302, grad norm 1.729826, param norm 1451.915283, tps 927.481680, length mean/std 2.000000/0.000000
INFO:root:epoch 45, iter 800, cost 1.671953, exp_cost 1.599056, grad norm 19.788870, param norm 1452.160767, tps 927.738492, length mean/std 31.000000/0.000000
INFO:root:epoch 45, iter 1000, cost 1.486347, exp_cost 1.582661, grad norm 8.617850, param norm 1452.455444, tps 927.983119, length mean/std 13.000000/0.000000
INFO:root:epoch 45, iter 1200, cost 1.511345, exp_cost 1.593808, grad norm 12.984987, param norm 1452.783691, tps 928.265163, length mean/std 20.000000/0.000000
INFO:root:epoch 45, iter 1400, cost 1.663975, exp_cost 1.600911, grad norm 28.235781, param norm 1453.092773, tps 928.536623, length mean/std 42.000000/0.000000
INFO:root:epoch 45, iter 1600, cost 1.576126, exp_cost 1.589000, grad norm 24.828680, param norm 1453.348877, tps 928.809508, length mean/std 43.000000/0.000000
INFO:root:epoch 45, iter 1800, cost 1.495161, exp_cost 1.596496, grad norm 15.344823, param norm 1453.622070, tps 929.080602, length mean/std 26.000000/0.000000
INFO:root:epoch 45, iter 2000, cost 1.668228, exp_cost 1.594522, grad norm 21.705456, param norm 1453.985474, tps 929.366193, length mean/std 32.000000/0.000000
INFO:root:epoch 45, iter 2200, cost 1.606264, exp_cost 1.581583, grad norm 21.508301, param norm 1454.255249, tps 929.594173, length mean/std 35.000000/0.000000
INFO:root:epoch 45, iter 2400, cost 1.632725, exp_cost 1.592114, grad norm 5.182710, param norm 1454.552246, tps 929.864716, length mean/std 7.000000/0.000000
INFO:root:epoch 45, iter 2600, cost 1.577098, exp_cost 1.592917, grad norm 12.260279, param norm 1454.859741, tps 930.107290, length mean/std 17.000000/0.000000
INFO:root:epoch 45, iter 2800, cost 1.545230, exp_cost 1.594764, grad norm 13.250400, param norm 1455.184937, tps 930.364867, length mean/std 21.000000/0.000000
INFO:root:epoch 45, iter 3000, cost 1.446381, exp_cost 1.591350, grad norm 8.852122, param norm 1455.428711, tps 930.590346, length mean/std 13.000000/0.000000
INFO:root:epoch 45, iter 3200, cost 1.761799, exp_cost 1.591635, grad norm 4.326737, param norm 1455.691162, tps 930.833969, length mean/std 5.000000/0.000000
INFO:root:epoch 45, iter 3400, cost 1.562131, exp_cost 1.595256, grad norm 10.596076, param norm 1455.944092, tps 931.037250, length mean/std 17.000000/0.000000
INFO:root:epoch 45, iter 3600, cost 1.565252, exp_cost 1.578711, grad norm 14.135194, param norm 1456.252441, tps 931.269935, length mean/std 21.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 80110559 get requests, put_count=80110549 evicted_count=23000 eviction_rate=0.000287103 and unsatisfied allocation rate=0.000293245
INFO:root:epoch 45, iter 3800, cost 1.490695, exp_cost 1.593688, grad norm 11.102074, param norm 1456.545654, tps 931.499576, length mean/std 17.000000/0.000000
INFO:root:epoch 45, iter 4000, cost 2.299985, exp_cost 1.599291, grad norm 1.757301, param norm 1456.808838, tps 931.747220, length mean/std 2.000000/0.000000
INFO:root:epoch 45, iter 4200, cost 1.463242, exp_cost 1.595432, grad norm 11.202806, param norm 1457.166992, tps 932.008922, length mean/std 17.000000/0.000000
INFO:root:epoch 45, iter 4400, cost 1.531950, exp_cost 1.584627, grad norm 7.414696, param norm 1457.453735, tps 932.247014, length mean/std 11.000000/0.000000
INFO:root:epoch 45, iter 4600, cost 1.580151, exp_cost 1.578467, grad norm 14.379194, param norm 1457.815186, tps 932.484059, length mean/std 22.000000/0.000000
INFO:root:epoch 45, iter 4800, cost 1.618061, exp_cost 1.593463, grad norm 11.089566, param norm 1458.061157, tps 932.713398, length mean/std 16.000000/0.000000
INFO:root:epoch 45, iter 5000, cost 1.524072, exp_cost 1.593489, grad norm 8.638031, param norm 1458.384766, tps 932.985524, length mean/std 13.000000/0.000000
INFO:root:epoch 45, iter 5200, cost 1.637021, exp_cost 1.594743, grad norm 13.670564, param norm 1458.668945, tps 933.234204, length mean/std 21.000000/0.000000
INFO:root:epoch 45, iter 5400, cost 2.446386, exp_cost 1.600242, grad norm 1.853065, param norm 1458.987427, tps 933.458855, length mean/std 2.000000/0.000000
INFO:root:epoch 45, iter 5600, cost 1.636466, exp_cost 1.579048, grad norm 4.363281, param norm 1459.188843, tps 933.649950, length mean/std 6.000000/0.000000
INFO:root:epoch 45, iter 5800, cost 1.592627, exp_cost 1.599103, grad norm 22.463451, param norm 1459.530518, tps 933.883245, length mean/std 38.000000/0.000000
INFO:root:epoch 45, iter 6000, cost 1.683340, exp_cost 1.593271, grad norm 23.272509, param norm 1459.791382, tps 934.108873, length mean/std 37.000000/0.000000
INFO:root:epoch 45, iter 6200, cost 1.500552, exp_cost 1.583926, grad norm 13.899063, param norm 1460.026489, tps 934.331653, length mean/std 22.000000/0.000000
INFO:root:epoch 45, iter 6400, cost 1.632266, exp_cost 1.600027, grad norm 3.123893, param norm 1460.343628, tps 934.529641, length mean/std 4.000000/0.000000
INFO:root:epoch 45, iter 6600, cost 1.550335, exp_cost 1.588052, grad norm 14.719263, param norm 1460.609863, tps 934.758930, length mean/std 22.000000/0.000000
INFO:root:epoch 45, iter 6800, cost 1.610382, exp_cost 1.595419, grad norm 18.046736, param norm 1460.916016, tps 934.952982, length mean/std 29.000000/0.000000
INFO:root:epoch 45, iter 7000, cost 1.618112, exp_cost 1.603711, grad norm 18.137899, param norm 1461.218140, tps 935.199072, length mean/std 29.000000/0.000000
INFO:root:epoch 45, iter 7200, cost 1.674248, exp_cost 1.595847, grad norm 17.685795, param norm 1461.448486, tps 935.427895, length mean/std 26.000000/0.000000
INFO:root:epoch 45, iter 7400, cost 1.528438, exp_cost 1.601448, grad norm 13.686099, param norm 1461.760254, tps 935.685682, length mean/std 22.000000/0.000000
INFO:root:epoch 45, iter 7600, cost 1.635724, exp_cost 1.600818, grad norm 19.178009, param norm 1462.027710, tps 935.874183, length mean/std 28.000000/0.000000
INFO:root:epoch 45, iter 7800, cost 2.000671, exp_cost 1.593871, grad norm 1.740660, param norm 1462.321533, tps 936.057083, length mean/std 2.000000/0.000000
INFO:root:epoch 45, iter 8000, cost 1.610099, exp_cost 1.598499, grad norm 11.834966, param norm 1462.593750, tps 936.279546, length mean/std 18.000000/0.000000
INFO:root:epoch 45, iter 8200, cost 1.612288, exp_cost 1.602814, grad norm 12.661570, param norm 1462.952759, tps 936.487814, length mean/std 19.000000/0.000000
INFO:root:epoch 45, iter 8400, cost 1.498586, exp_cost 1.588799, grad norm 7.969946, param norm 1463.211670, tps 936.697090, length mean/std 12.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 100009181 get requests, put_count=100009172 evicted_count=33000 eviction_rate=0.00032997 and unsatisfied allocation rate=0.000334879
INFO:root:epoch 45, iter 8600, cost 1.668622, exp_cost 1.607429, grad norm 5.486666, param norm 1463.619751, tps 936.934902, length mean/std 7.000000/0.000000
INFO:root:epoch 45, iter 8800, cost 1.571011, exp_cost 1.595046, grad norm 7.093948, param norm 1463.966553, tps 937.100293, length mean/std 10.000000/0.000000
INFO:root:epoch 45, iter 9000, cost 1.862829, exp_cost 1.597811, grad norm 2.586227, param norm 1464.210205, tps 937.273635, length mean/std 3.000000/0.000000
INFO:root:epoch 45, iter 9200, cost 1.673343, exp_cost 1.597600, grad norm 25.806261, param norm 1464.488159, tps 937.487326, length mean/std 42.000000/0.000000
INFO:root:epoch 45, iter 9400, cost 1.295953, exp_cost 1.590027, grad norm 7.910022, param norm 1464.798584, tps 937.702404, length mean/std 13.000000/0.000000
INFO:root:epoch 45, iter 9600, cost 1.469270, exp_cost 1.596951, grad norm 6.954924, param norm 1465.069580, tps 937.890860, length mean/std 10.000000/0.000000
INFO:root:epoch 45, iter 9800, cost 1.805254, exp_cost 1.601563, grad norm 8.142347, param norm 1465.307983, tps 938.077615, length mean/std 10.000000/0.000000
INFO:root:epoch 45, iter 10000, cost 1.639173, exp_cost 1.605977, grad norm 18.122084, param norm 1465.629761, tps 938.252868, length mean/std 28.000000/0.000000
INFO:root:epoch 45, iter 10200, cost 1.670825, exp_cost 1.610685, grad norm 5.999413, param norm 1465.977173, tps 938.460536, length mean/std 8.000000/0.000000
INFO:root:epoch 45, iter 10400, cost 2.830101, exp_cost 1.606316, grad norm 0.787706, param norm 1466.203003, tps 938.665666, length mean/std 1.000000/0.000000
INFO:root:epoch 45, iter 10600, cost 1.741914, exp_cost 1.595838, grad norm 5.524762, param norm 1466.537354, tps 938.903214, length mean/std 7.000000/0.000000
INFO:root:epoch 45, iter 10800, cost 1.660838, exp_cost 1.601840, grad norm 5.924513, param norm 1466.812012, tps 939.111111, length mean/std 7.000000/0.000000
INFO:root:epoch 45, iter 11000, cost 1.692731, exp_cost 1.604547, grad norm 19.792492, param norm 1467.125244, tps 939.338781, length mean/std 33.000000/0.000000
INFO:root:epoch 45, iter 11200, cost 1.448192, exp_cost 1.590843, grad norm 11.201680, param norm 1467.320801, tps 939.553428, length mean/std 18.000000/0.000000
INFO:root:epoch 45, iter 11400, cost 1.463990, exp_cost 1.611454, grad norm 9.045822, param norm 1467.658203, tps 939.748950, length mean/std 14.000000/0.000000
INFO:root:epoch 45, iter 11600, cost 1.671868, exp_cost 1.596094, grad norm 11.395945, param norm 1467.883545, tps 939.938459, length mean/std 16.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 113744022 get requests, put_count=113743928 evicted_count=43000 eviction_rate=0.000378042 and unsatisfied allocation rate=0.000383106
INFO:root:epoch 45, iter 11800, cost 1.603906, exp_cost 1.601794, grad norm 21.807699, param norm 1468.217407, tps 940.152655, length mean/std 33.000000/0.000000
INFO:root:epoch 45, iter 12000, cost 1.474468, exp_cost 1.598669, grad norm 13.145790, param norm 1468.491699, tps 940.353534, length mean/std 22.000000/0.000000
INFO:root:epoch 45, iter 12200, cost 1.513089, exp_cost 1.600710, grad norm 13.475528, param norm 1468.762939, tps 940.526388, length mean/std 21.000000/0.000000
INFO:root:epoch 45, iter 12400, cost 1.731127, exp_cost 1.602344, grad norm 5.736925, param norm 1469.056763, tps 940.695111, length mean/std 7.000000/0.000000
INFO:root:Epoch 45 Validation cost: 1.700641 time: 7025.419266
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-45 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-45 is not in all_model_checkpoint_paths. Manually adding it.
INFO:root:epoch 46, iter 200, cost 1.698395, exp_cost 1.576697, grad norm 5.509142, param norm 1469.544434, tps 928.073693, length mean/std 7.000000/0.000000
INFO:root:epoch 46, iter 400, cost 1.591865, exp_cost 1.588995, grad norm 19.318933, param norm 1469.882446, tps 928.285955, length mean/std 32.000000/0.000000
INFO:root:epoch 46, iter 600, cost 1.597246, exp_cost 1.579118, grad norm 16.957886, param norm 1470.185913, tps 928.500012, length mean/std 24.000000/0.000000
INFO:root:epoch 46, iter 800, cost 1.448792, exp_cost 1.585729, grad norm 2.974462, param norm 1470.480591, tps 928.725568, length mean/std 4.000000/0.000000
INFO:root:epoch 46, iter 1000, cost 1.607991, exp_cost 1.587938, grad norm 10.805574, param norm 1470.762085, tps 928.987746, length mean/std 15.000000/0.000000
INFO:root:epoch 46, iter 1200, cost 1.853448, exp_cost 1.590258, grad norm 4.668862, param norm 1470.922119, tps 929.213209, length mean/std 5.000000/0.000000
INFO:root:epoch 46, iter 1400, cost 1.569390, exp_cost 1.585449, grad norm 12.489555, param norm 1471.186035, tps 929.431693, length mean/std 16.000000/0.000000
INFO:root:epoch 46, iter 1600, cost 1.555942, exp_cost 1.581319, grad norm 12.705364, param norm 1471.426147, tps 929.630512, length mean/std 19.000000/0.000000
INFO:root:epoch 46, iter 1800, cost 1.503998, exp_cost 1.569120, grad norm 9.273897, param norm 1471.791992, tps 929.861810, length mean/std 14.000000/0.000000
INFO:root:epoch 46, iter 2000, cost 1.700800, exp_cost 1.597753, grad norm 23.869265, param norm 1472.015137, tps 930.073641, length mean/std 37.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 126840205 get requests, put_count=126840115 evicted_count=53000 eviction_rate=0.000417849 and unsatisfied allocation rate=0.000422358
INFO:root:epoch 46, iter 2200, cost 2.124068, exp_cost 1.595800, grad norm 2.812075, param norm 1472.280518, tps 930.282332, length mean/std 3.000000/0.000000
INFO:root:epoch 46, iter 2400, cost 1.567063, exp_cost 1.594032, grad norm 16.085859, param norm 1472.550781, tps 930.477916, length mean/std 25.000000/0.000000
INFO:root:epoch 46, iter 2600, cost 1.853318, exp_cost 1.584183, grad norm 3.785394, param norm 1472.773438, tps 930.697590, length mean/std 4.000000/0.000000
INFO:root:epoch 46, iter 2800, cost 1.439014, exp_cost 1.592454, grad norm 9.766280, param norm 1473.054688, tps 930.888090, length mean/std 15.000000/0.000000
INFO:root:epoch 46, iter 3000, cost 1.467949, exp_cost 1.583654, grad norm 12.258733, param norm 1473.333862, tps 931.094026, length mean/std 21.000000/0.000000
INFO:root:epoch 46, iter 3200, cost 1.628150, exp_cost 1.597073, grad norm 20.262598, param norm 1473.561646, tps 931.298387, length mean/std 32.031250/0.173993
INFO:root:epoch 46, iter 3400, cost 1.551252, exp_cost 1.586538, grad norm 17.989473, param norm 1473.841675, tps 931.488941, length mean/std 27.000000/0.000000
INFO:root:epoch 46, iter 3600, cost 1.599270, exp_cost 1.592162, grad norm 17.802729, param norm 1474.136475, tps 931.730954, length mean/std 29.000000/0.000000
INFO:root:epoch 46, iter 3800, cost 1.536681, exp_cost 1.586386, grad norm 14.203518, param norm 1474.428101, tps 931.928561, length mean/std 24.000000/0.000000
INFO:root:epoch 46, iter 4000, cost 1.521932, exp_cost 1.591917, grad norm 15.480850, param norm 1474.718262, tps 932.148108, length mean/std 26.000000/0.000000
INFO:root:epoch 46, iter 4200, cost 1.529495, exp_cost 1.577769, grad norm 11.987084, param norm 1475.057007, tps 932.352168, length mean/std 18.000000/0.000000
INFO:root:epoch 46, iter 4400, cost 1.566237, exp_cost 1.593340, grad norm 5.040350, param norm 1475.334839, tps 932.575150, length mean/std 7.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 136761830 get requests, put_count=136761802 evicted_count=63000 eviction_rate=0.000460655 and unsatisfied allocation rate=0.000464384
INFO:root:epoch 46, iter 4600, cost 1.674283, exp_cost 1.596031, grad norm 7.088842, param norm 1475.604492, tps 932.804376, length mean/std 10.000000/0.000000
INFO:root:epoch 46, iter 4800, cost 1.904470, exp_cost 1.596710, grad norm 5.183888, param norm 1475.885132, tps 932.999257, length mean/std 6.000000/0.000000
INFO:root:epoch 46, iter 5000, cost 1.752137, exp_cost 1.592700, grad norm 3.338056, param norm 1476.114380, tps 933.197757, length mean/std 4.000000/0.000000
INFO:root:epoch 46, iter 5200, cost 1.532272, exp_cost 1.598225, grad norm 13.485842, param norm 1476.481323, tps 933.433908, length mean/std 20.000000/0.000000
INFO:root:epoch 46, iter 5400, cost 1.698834, exp_cost 1.599827, grad norm 23.818537, param norm 1476.664185, tps 933.631264, length mean/std 37.000000/0.000000
INFO:root:epoch 46, iter 5600, cost 1.530568, exp_cost 1.595276, grad norm 8.196605, param norm 1476.919312, tps 933.852252, length mean/std 12.000000/0.000000
INFO:root:epoch 46, iter 5800, cost 1.513397, exp_cost 1.593621, grad norm 14.994571, param norm 1477.194702, tps 934.029038, length mean/std 24.000000/0.000000
INFO:root:epoch 46, iter 6000, cost 1.546349, exp_cost 1.584710, grad norm 4.969898, param norm 1477.459351, tps 934.227427, length mean/std 7.000000/0.000000
INFO:root:epoch 46, iter 6200, cost 1.466948, exp_cost 1.582161, grad norm 13.612331, param norm 1477.750610, tps 934.405454, length mean/std 23.000000/0.000000
INFO:root:epoch 46, iter 6400, cost 1.434642, exp_cost 1.590063, grad norm 14.988907, param norm 1478.026123, tps 934.602438, length mean/std 27.000000/0.000000
INFO:root:epoch 46, iter 6600, cost 1.662066, exp_cost 1.591103, grad norm 21.159876, param norm 1478.361084, tps 934.791085, length mean/std 35.000000/0.000000
INFO:root:epoch 46, iter 6800, cost 1.697631, exp_cost 1.588505, grad norm 19.274956, param norm 1478.618896, tps 934.983428, length mean/std 30.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 147316177 get requests, put_count=147316101 evicted_count=73000 eviction_rate=0.000495533 and unsatisfied allocation rate=0.000499321
INFO:root:epoch 46, iter 7000, cost 1.654403, exp_cost 1.586608, grad norm 4.738779, param norm 1478.987183, tps 935.189385, length mean/std 6.000000/0.000000
INFO:root:epoch 46, iter 7200, cost 1.537820, exp_cost 1.601970, grad norm 6.958623, param norm 1479.229980, tps 935.399804, length mean/std 10.000000/0.000000
INFO:root:epoch 46, iter 7400, cost 1.607934, exp_cost 1.605427, grad norm 11.209286, param norm 1479.588257, tps 935.599219, length mean/std 16.000000/0.000000
INFO:root:epoch 46, iter 7600, cost 1.797506, exp_cost 1.594323, grad norm 3.383938, param norm 1479.828735, tps 935.801035, length mean/std 4.000000/0.000000
INFO:root:epoch 46, iter 7800, cost 1.662000, exp_cost 1.592296, grad norm 7.967540, param norm 1480.120972, tps 935.983493, length mean/std 11.000000/0.000000
INFO:root:epoch 46, iter 8000, cost 1.704250, exp_cost 1.598708, grad norm 3.392379, param norm 1480.440186, tps 936.167061, length mean/std 4.000000/0.000000
INFO:root:epoch 46, iter 8200, cost 1.666295, exp_cost 1.588697, grad norm 25.675940, param norm 1480.708984, tps 936.326649, length mean/std 41.000000/0.000000
INFO:root:epoch 46, iter 8400, cost 1.598845, exp_cost 1.596404, grad norm 9.873007, param norm 1481.031494, tps 936.523271, length mean/std 14.000000/0.000000
INFO:root:epoch 46, iter 8600, cost 1.727714, exp_cost 1.605569, grad norm 3.350497, param norm 1481.274658, tps 936.712683, length mean/std 4.000000/0.000000
INFO:root:epoch 46, iter 8800, cost 1.479384, exp_cost 1.594931, grad norm 7.595282, param norm 1481.480957, tps 936.902318, length mean/std 11.000000/0.000000
INFO:root:epoch 46, iter 9000, cost 1.699472, exp_cost 1.588758, grad norm 21.239922, param norm 1481.777344, tps 937.125283, length mean/std 31.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 156203194 get requests, put_count=156203091 evicted_count=83000 eviction_rate=0.00053136 and unsatisfied allocation rate=0.000535104
INFO:root:epoch 46, iter 9200, cost 1.429335, exp_cost 1.595487, grad norm 9.463846, param norm 1482.021484, tps 937.318390, length mean/std 16.000000/0.000000
INFO:root:epoch 46, iter 9400, cost 1.714348, exp_cost 1.608949, grad norm 5.537993, param norm 1482.362793, tps 937.495304, length mean/std 7.000000/0.000000
INFO:root:epoch 46, iter 9600, cost 1.598127, exp_cost 1.602153, grad norm 5.178142, param norm 1482.569702, tps 937.715000, length mean/std 7.000000/0.000000
INFO:root:epoch 46, iter 9800, cost 1.541261, exp_cost 1.597250, grad norm 9.550706, param norm 1482.846802, tps 937.892121, length mean/std 14.000000/0.000000
INFO:root:epoch 46, iter 10000, cost 1.783120, exp_cost 1.607517, grad norm 6.279739, param norm 1483.166138, tps 938.095323, length mean/std 8.000000/0.000000
INFO:root:epoch 46, iter 10200, cost 1.597265, exp_cost 1.610881, grad norm 21.284061, param norm 1483.401123, tps 938.277231, length mean/std 35.000000/0.000000
INFO:root:epoch 46, iter 10400, cost 1.772414, exp_cost 1.603334, grad norm 5.265141, param norm 1483.636963, tps 938.452160, length mean/std 6.000000/0.000000
INFO:root:epoch 46, iter 10600, cost 1.506121, exp_cost 1.590184, grad norm 12.715301, param norm 1483.933960, tps 938.606332, length mean/std 18.000000/0.000000
INFO:root:epoch 46, iter 10800, cost 1.504813, exp_cost 1.594251, grad norm 9.150191, param norm 1484.223877, tps 938.777998, length mean/std 13.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 163913887 get requests, put_count=163913823 evicted_count=93000 eviction_rate=0.000567371 and unsatisfied allocation rate=0.000570702
INFO:root:epoch 46, iter 11000, cost 1.730915, exp_cost 1.601052, grad norm 25.677094, param norm 1484.512207, tps 938.957455, length mean/std 42.000000/0.000000
INFO:root:epoch 46, iter 11200, cost 1.633865, exp_cost 1.604772, grad norm 10.345338, param norm 1484.736206, tps 939.117252, length mean/std 15.000000/0.000000
INFO:root:epoch 46, iter 11400, cost 1.626402, exp_cost 1.606384, grad norm 20.210783, param norm 1485.050903, tps 939.282112, length mean/std 31.000000/0.000000
INFO:root:epoch 46, iter 11600, cost 1.646634, exp_cost 1.599925, grad norm 15.203006, param norm 1485.330322, tps 939.479743, length mean/std 23.000000/0.000000
INFO:root:epoch 46, iter 11800, cost 1.771029, exp_cost 1.604957, grad norm 4.835298, param norm 1485.590088, tps 939.651832, length mean/std 6.000000/0.000000
INFO:root:epoch 46, iter 12000, cost 1.510958, exp_cost 1.610942, grad norm 4.840088, param norm 1485.838867, tps 939.823087, length mean/std 7.000000/0.000000
INFO:root:epoch 46, iter 12200, cost 1.667898, exp_cost 1.605676, grad norm 24.409100, param norm 1486.184570, tps 940.011265, length mean/std 41.000000/0.000000
INFO:root:epoch 46, iter 12400, cost 1.596896, exp_cost 1.611038, grad norm 8.967392, param norm 1486.511719, tps 940.188072, length mean/std 13.000000/0.000000
INFO:root:Epoch 46 Validation cost: 1.701929 time: 7116.276683
INFO:root:Annealing learning rate by 0.950000
INFO:root:epoch 47, iter 200, cost 1.524529, exp_cost 1.589962, grad norm 13.794588, param norm 1469.497925, tps 929.129378, length mean/std 24.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 173194436 get requests, put_count=173194280 evicted_count=103000 eviction_rate=0.000594708 and unsatisfied allocation rate=0.000598391
INFO:root:epoch 47, iter 400, cost 1.640214, exp_cost 1.591871, grad norm 20.854185, param norm 1469.769165, tps 929.348812, length mean/std 31.000000/0.000000
INFO:root:epoch 47, iter 600, cost 1.778890, exp_cost 1.573102, grad norm 4.413472, param norm 1470.073120, tps 929.558656, length mean/std 5.000000/0.000000
INFO:root:epoch 47, iter 800, cost 1.464963, exp_cost 1.578527, grad norm 11.098892, param norm 1470.500366, tps 929.751243, length mean/std 17.000000/0.000000
INFO:root:epoch 47, iter 1000, cost 1.569265, exp_cost 1.584871, grad norm 14.756903, param norm 1470.687988, tps 929.930822, length mean/std 20.000000/0.000000
INFO:root:epoch 47, iter 1200, cost 1.640125, exp_cost 1.590903, grad norm 24.665100, param norm 1470.973511, tps 930.137740, length mean/std 39.000000/0.000000
INFO:root:epoch 47, iter 1400, cost 1.624304, exp_cost 1.579282, grad norm 17.043203, param norm 1471.219849, tps 930.341057, length mean/std 27.000000/0.000000
INFO:root:epoch 47, iter 1600, cost 1.499834, exp_cost 1.584306, grad norm 6.719177, param norm 1471.491211, tps 930.501959, length mean/std 10.000000/0.000000
INFO:root:epoch 47, iter 1800, cost 1.537549, exp_cost 1.586871, grad norm 14.591870, param norm 1471.786621, tps 930.717200, length mean/std 22.000000/0.000000
INFO:root:epoch 47, iter 2000, cost 2.506241, exp_cost 1.581235, grad norm 2.594176, param norm 1472.070190, tps 930.883780, length mean/std 2.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 181378544 get requests, put_count=181378417 evicted_count=113000 eviction_rate=0.000623007 and unsatisfied allocation rate=0.000626364
INFO:root:epoch 47, iter 2200, cost 1.526459, exp_cost 1.590734, grad norm 6.882943, param norm 1472.399536, tps 931.086974, length mean/std 10.000000/0.000000
INFO:root:epoch 47, iter 2400, cost 1.596959, exp_cost 1.592332, grad norm 5.429183, param norm 1472.647705, tps 931.265869, length mean/std 8.000000/0.000000
INFO:root:epoch 47, iter 2600, cost 2.702092, exp_cost 1.592407, grad norm 0.713714, param norm 1472.889038, tps 931.449155, length mean/std 1.000000/0.000000
INFO:root:epoch 47, iter 2800, cost 1.646423, exp_cost 1.593136, grad norm 23.767570, param norm 1473.205200, tps 931.613936, length mean/std 36.000000/0.000000
INFO:root:epoch 47, iter 3000, cost 1.561464, exp_cost 1.586898, grad norm 10.933443, param norm 1473.352905, tps 931.778154, length mean/std 15.000000/0.000000
INFO:root:epoch 47, iter 3200, cost 1.516732, exp_cost 1.587915, grad norm 13.813456, param norm 1473.638916, tps 931.962815, length mean/std 21.000000/0.000000
INFO:root:epoch 47, iter 3400, cost 1.645341, exp_cost 1.586074, grad norm 7.402097, param norm 1473.976196, tps 932.140885, length mean/std 10.000000/0.000000
INFO:root:epoch 47, iter 3600, cost 2.002669, exp_cost 1.576914, grad norm 4.620668, param norm 1474.260498, tps 932.311523, length mean/std 5.000000/0.000000
INFO:root:epoch 47, iter 3800, cost 1.544005, exp_cost 1.583262, grad norm 16.868597, param norm 1474.553467, tps 932.505135, length mean/std 29.000000/0.000000
INFO:root:epoch 47, iter 4000, cost 1.559360, exp_cost 1.584107, grad norm 18.021357, param norm 1474.850220, tps 932.702587, length mean/std 28.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 189260637 get requests, put_count=189260605 evicted_count=123000 eviction_rate=0.000649898 and unsatisfied allocation rate=0.000652613
INFO:root:epoch 47, iter 4200, cost 1.576631, exp_cost 1.592416, grad norm 15.499929, param norm 1475.197144, tps 932.886305, length mean/std 24.000000/0.000000
INFO:root:epoch 47, iter 4400, cost 1.709132, exp_cost 1.597166, grad norm 6.129990, param norm 1475.435913, tps 933.044934, length mean/std 8.000000/0.000000
INFO:root:epoch 47, iter 4600, cost 1.656017, exp_cost 1.593408, grad norm 19.434679, param norm 1475.737549, tps 933.206112, length mean/std 29.000000/0.000000
INFO:root:epoch 47, iter 4800, cost 1.649627, exp_cost 1.595384, grad norm 18.894316, param norm 1475.984131, tps 933.387303, length mean/std 30.000000/0.000000
INFO:root:epoch 47, iter 5000, cost 1.635161, exp_cost 1.610009, grad norm 4.965672, param norm 1476.276855, tps 933.582991, length mean/std 7.000000/0.000000
INFO:root:epoch 47, iter 5200, cost 1.658549, exp_cost 1.598756, grad norm 8.046805, param norm 1476.611206, tps 933.762684, length mean/std 11.000000/0.000000
INFO:root:epoch 47, iter 5400, cost 1.623348, exp_cost 1.583206, grad norm 27.528875, param norm 1476.842896, tps 933.936567, length mean/std 45.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 194944428 get requests, put_count=194944301 evicted_count=133000 eviction_rate=0.000682246 and unsatisfied allocation rate=0.00068537
INFO:root:epoch 47, iter 5600, cost 1.494401, exp_cost 1.596363, grad norm 11.951234, param norm 1477.139404, tps 934.115776, length mean/std 20.000000/0.000000
INFO:root:epoch 47, iter 5800, cost 1.591095, exp_cost 1.600730, grad norm 16.940886, param norm 1477.355957, tps 934.317592, length mean/std 28.000000/0.000000
INFO:root:epoch 47, iter 6000, cost 1.411181, exp_cost 1.589096, grad norm 6.836460, param norm 1477.622803, tps 934.498022, length mean/std 10.000000/0.000000
INFO:root:epoch 47, iter 6200, cost 1.581294, exp_cost 1.588305, grad norm 7.914637, param norm 1477.868530, tps 934.666383, length mean/std 10.000000/0.000000
INFO:root:epoch 47, iter 6400, cost 1.565274, exp_cost 1.592168, grad norm 15.409217, param norm 1478.111084, tps 934.837507, length mean/std 25.000000/0.000000
INFO:root:epoch 47, iter 6600, cost 1.578658, exp_cost 1.602276, grad norm 16.507591, param norm 1478.401367, tps 935.022535, length mean/std 26.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 200490393 get requests, put_count=200490259 evicted_count=143000 eviction_rate=0.000713252 and unsatisfied allocation rate=0.000716324
INFO:root:epoch 47, iter 6800, cost 1.559383, exp_cost 1.594473, grad norm 9.470485, param norm 1478.617310, tps 935.197877, length mean/std 13.000000/0.000000
INFO:root:epoch 47, iter 7000, cost 1.470060, exp_cost 1.595410, grad norm 8.442385, param norm 1478.817627, tps 935.388327, length mean/std 13.000000/0.000000
INFO:root:epoch 47, iter 7200, cost 1.674827, exp_cost 1.584786, grad norm 16.360289, param norm 1479.160522, tps 935.551932, length mean/std 25.000000/0.000000
INFO:root:epoch 47, iter 7400, cost 1.669119, exp_cost 1.592339, grad norm 6.249383, param norm 1479.435913, tps 935.728431, length mean/std 8.000000/0.000000
INFO:root:epoch 47, iter 7600, cost 1.789821, exp_cost 1.593973, grad norm 4.894849, param norm 1479.808838, tps 935.909687, length mean/std 6.000000/0.000000
INFO:root:epoch 47, iter 7800, cost 2.354789, exp_cost 1.600067, grad norm 1.936594, param norm 1480.033569, tps 936.060402, length mean/std 2.000000/0.000000
INFO:root:epoch 47, iter 8000, cost 2.237568, exp_cost 1.598098, grad norm 2.943760, param norm 1480.232788, tps 936.203678, length mean/std 3.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 206248859 get requests, put_count=206248830 evicted_count=153000 eviction_rate=0.000741822 and unsatisfied allocation rate=0.0007443
INFO:root:epoch 47, iter 8200, cost 1.535536, exp_cost 1.598779, grad norm 20.062454, param norm 1480.490967, tps 936.384481, length mean/std 33.000000/0.000000
INFO:root:epoch 47, iter 8400, cost 1.561242, exp_cost 1.599931, grad norm 17.050848, param norm 1480.707397, tps 936.568850, length mean/std 29.000000/0.000000
INFO:root:epoch 47, iter 8600, cost 1.709262, exp_cost 1.593228, grad norm 24.732744, param norm 1481.004150, tps 936.777973, length mean/std 39.000000/0.000000
INFO:root:epoch 47, iter 8800, cost 1.664349, exp_cost 1.601611, grad norm 15.075527, param norm 1481.359375, tps 936.950749, length mean/std 24.000000/0.000000
INFO:root:epoch 47, iter 9000, cost 1.678049, exp_cost 1.607495, grad norm 25.913147, param norm 1481.681396, tps 937.155827, length mean/std 47.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 210927711 get requests, put_count=210927667 evicted_count=163000 eviction_rate=0.000772777 and unsatisfied allocation rate=0.00077527
INFO:root:epoch 47, iter 9200, cost 1.437619, exp_cost 1.600709, grad norm 8.465705, param norm 1481.903809, tps 937.331462, length mean/std 13.000000/0.000000
INFO:root:epoch 47, iter 9400, cost 1.633679, exp_cost 1.602464, grad norm 6.173546, param norm 1482.143677, tps 937.476558, length mean/std 8.000000/0.000000
INFO:root:epoch 47, iter 9600, cost 1.572703, exp_cost 1.597811, grad norm 13.432110, param norm 1482.440796, tps 937.663690, length mean/std 21.000000/0.000000
INFO:root:epoch 47, iter 9800, cost 1.641893, exp_cost 1.598805, grad norm 16.843832, param norm 1482.699341, tps 937.847104, length mean/std 27.000000/0.000000
INFO:root:epoch 47, iter 10000, cost 1.760744, exp_cost 1.595085, grad norm 2.400890, param norm 1482.964355, tps 938.021492, length mean/std 3.000000/0.000000
INFO:root:epoch 47, iter 10200, cost 1.654270, exp_cost 1.602948, grad norm 21.488680, param norm 1483.195435, tps 938.193559, length mean/std 34.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 216648423 get requests, put_count=216648367 evicted_count=173000 eviction_rate=0.000798529 and unsatisfied allocation rate=0.000801012
INFO:root:epoch 47, iter 10400, cost 1.745749, exp_cost 1.607188, grad norm 5.041995, param norm 1483.519409, tps 938.363689, length mean/std 6.000000/0.000000
INFO:root:epoch 47, iter 10600, cost 1.513456, exp_cost 1.598340, grad norm 9.241855, param norm 1483.808105, tps 938.517415, length mean/std 13.000000/0.000000
INFO:root:epoch 47, iter 10800, cost 1.706340, exp_cost 1.604127, grad norm 6.277178, param norm 1484.055054, tps 938.707840, length mean/std 8.000000/0.000000
INFO:root:epoch 47, iter 11000, cost 1.535611, exp_cost 1.611838, grad norm 9.630951, param norm 1484.443970, tps 938.873791, length mean/std 14.000000/0.000000
INFO:root:epoch 47, iter 11200, cost 1.538030, exp_cost 1.596713, grad norm 14.168626, param norm 1484.729492, tps 939.026230, length mean/std 24.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 220552773 get requests, put_count=220552554 evicted_count=183000 eviction_rate=0.000829734 and unsatisfied allocation rate=0.000832912
INFO:root:epoch 47, iter 11400, cost 1.640302, exp_cost 1.601531, grad norm 6.268829, param norm 1485.059448, tps 939.178791, length mean/std 9.000000/0.000000
INFO:root:epoch 47, iter 11600, cost 1.456212, exp_cost 1.590345, grad norm 9.292237, param norm 1485.340576, tps 939.343230, length mean/std 15.000000/0.000000
INFO:root:epoch 47, iter 11800, cost 1.647285, exp_cost 1.602009, grad norm 15.530137, param norm 1485.538940, tps 939.504654, length mean/std 22.000000/0.000000
INFO:root:epoch 47, iter 12000, cost 1.585028, exp_cost 1.590176, grad norm 8.817943, param norm 1485.833130, tps 939.650885, length mean/std 12.000000/0.000000
INFO:root:epoch 47, iter 12200, cost 1.513177, exp_cost 1.594175, grad norm 10.132067, param norm 1486.127319, tps 939.824089, length mean/std 14.000000/0.000000
INFO:root:epoch 47, iter 12400, cost 1.502576, exp_cost 1.598978, grad norm 10.127922, param norm 1486.378296, tps 939.963151, length mean/std 16.000000/0.000000
INFO:root:Epoch 47 Validation cost: 1.701633 time: 7210.403470
INFO:root:Annealing learning rate by 0.950000
INFO:root:epoch 48, iter 200, cost 1.452317, exp_cost 1.589948, grad norm 10.357210, param norm 1469.543091, tps 930.202853, length mean/std 17.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 227745232 get requests, put_count=227744910 evicted_count=193000 eviction_rate=0.000847439 and unsatisfied allocation rate=0.000850968
INFO:root:epoch 48, iter 400, cost 2.153084, exp_cost 1.581436, grad norm 1.855534, param norm 1469.781250, tps 930.375096, length mean/std 2.000000/0.000000
INFO:root:epoch 48, iter 600, cost 1.600534, exp_cost 1.586020, grad norm 21.051447, param norm 1469.981079, tps 930.538219, length mean/std 36.000000/0.000000
INFO:root:epoch 48, iter 800, cost 2.234020, exp_cost 1.585161, grad norm 2.204698, param norm 1470.251953, tps 930.744010, length mean/std 2.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 230888188 get requests, put_count=230888098 evicted_count=203000 eviction_rate=0.000879214 and unsatisfied allocation rate=0.000881691
INFO:root:epoch 48, iter 1000, cost 1.600198, exp_cost 1.579650, grad norm 2.970739, param norm 1470.584473, tps 930.926951, length mean/std 4.000000/0.000000
INFO:root:epoch 48, iter 1200, cost 1.656984, exp_cost 1.572249, grad norm 3.866636, param norm 1470.809326, tps 931.102667, length mean/std 5.000000/0.000000
INFO:root:epoch 48, iter 1400, cost 2.250923, exp_cost 1.583098, grad norm 1.868336, param norm 1471.067017, tps 931.254260, length mean/std 2.000000/0.000000
INFO:root:epoch 48, iter 1600, cost 1.599174, exp_cost 1.575904, grad norm 10.721158, param norm 1471.342896, tps 931.438192, length mean/std 15.000000/0.000000
INFO:root:epoch 48, iter 1800, cost 1.389683, exp_cost 1.589604, grad norm 9.407775, param norm 1471.665527, tps 931.603252, length mean/std 16.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 235022319 get requests, put_count=235022228 evicted_count=213000 eviction_rate=0.000906297 and unsatisfied allocation rate=0.000908735
INFO:root:epoch 48, iter 2000, cost 1.637318, exp_cost 1.590493, grad norm 5.580990, param norm 1471.904907, tps 931.766046, length mean/std 8.000000/0.000000
INFO:root:epoch 48, iter 2200, cost 1.554220, exp_cost 1.590565, grad norm 13.341527, param norm 1472.153687, tps 931.926918, length mean/std 20.000000/0.000000
INFO:root:epoch 48, iter 2400, cost 1.510579, exp_cost 1.592710, grad norm 8.916356, param norm 1472.397339, tps 932.116572, length mean/std 13.000000/0.000000
INFO:root:epoch 48, iter 2600, cost 1.512062, exp_cost 1.579556, grad norm 6.656159, param norm 1472.674805, tps 932.293416, length mean/std 10.000000/0.000000
INFO:root:epoch 48, iter 2800, cost 1.639522, exp_cost 1.577966, grad norm 7.003201, param norm 1472.979248, tps 932.464258, length mean/std 9.000000/0.000000
INFO:root:epoch 48, iter 3000, cost 1.500622, exp_cost 1.576518, grad norm 9.403207, param norm 1473.241455, tps 932.618893, length mean/std 15.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 240439082 get requests, put_count=240438912 evicted_count=223000 eviction_rate=0.000927471 and unsatisfied allocation rate=0.000930182
INFO:root:epoch 48, iter 3200, cost 1.567450, exp_cost 1.583743, grad norm 9.024334, param norm 1473.552979, tps 932.776825, length mean/std 14.000000/0.000000
INFO:root:epoch 48, iter 3400, cost 1.536253, exp_cost 1.580764, grad norm 4.487806, param norm 1473.876465, tps 932.936988, length mean/std 6.000000/0.000000
INFO:root:epoch 48, iter 3600, cost 1.650218, exp_cost 1.583194, grad norm 9.084160, param norm 1474.179199, tps 933.107712, length mean/std 13.000000/0.000000
INFO:root:epoch 48, iter 3800, cost 1.971235, exp_cost 1.584539, grad norm 2.726389, param norm 1474.510864, tps 933.287841, length mean/std 3.000000/0.000000
INFO:root:epoch 48, iter 4000, cost 1.537839, exp_cost 1.591727, grad norm 13.450020, param norm 1474.788574, tps 933.436102, length mean/std 21.000000/0.000000
INFO:root:epoch 48, iter 4200, cost 1.592236, exp_cost 1.604916, grad norm 13.529455, param norm 1475.042847, tps 933.595112, length mean/std 21.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 245167732 get requests, put_count=245167269 evicted_count=233000 eviction_rate=0.000950372 and unsatisfied allocation rate=0.000954224
INFO:root:epoch 48, iter 4400, cost 1.484855, exp_cost 1.592876, grad norm 8.052561, param norm 1475.375000, tps 933.756029, length mean/std 12.000000/0.000000
INFO:root:epoch 48, iter 4600, cost 1.537476, exp_cost 1.592101, grad norm 8.418534, param norm 1475.668091, tps 933.933486, length mean/std 12.000000/0.000000
INFO:root:epoch 48, iter 4800, cost 1.586071, exp_cost 1.596120, grad norm 10.130028, param norm 1475.885498, tps 934.112421, length mean/std 15.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 248183260 get requests, put_count=248183138 evicted_count=243000 eviction_rate=0.000979116 and unsatisfied allocation rate=0.000981549
INFO:root:epoch 48, iter 5000, cost 1.558021, exp_cost 1.602625, grad norm 9.792369, param norm 1476.161865, tps 934.271293, length mean/std 14.000000/0.000000
INFO:root:epoch 48, iter 5200, cost 1.584025, exp_cost 1.595609, grad norm 16.234718, param norm 1476.417236, tps 934.444339, length mean/std 25.000000/0.000000
INFO:root:epoch 48, iter 5400, cost 1.597894, exp_cost 1.592849, grad norm 9.919456, param norm 1476.696167, tps 934.611892, length mean/std 14.000000/0.000000
INFO:root:epoch 48, iter 5600, cost 1.531898, exp_cost 1.589904, grad norm 15.272178, param norm 1476.977783, tps 934.772172, length mean/std 22.000000/0.000000
INFO:root:epoch 48, iter 5800, cost 1.579602, exp_cost 1.594168, grad norm 20.687685, param norm 1477.286743, tps 934.947763, length mean/std 36.000000/0.000000
INFO:root:epoch 48, iter 6000, cost 1.665531, exp_cost 1.597949, grad norm 20.514227, param norm 1477.538452, tps 935.083270, length mean/std 31.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 253473255 get requests, put_count=253472912 evicted_count=253000 eviction_rate=0.000998134 and unsatisfied allocation rate=0.00100139
INFO:root:epoch 48, iter 6200, cost 2.215629, exp_cost 1.587573, grad norm 2.274934, param norm 1477.794922, tps 935.238369, length mean/std 2.000000/0.000000
INFO:root:epoch 48, iter 6400, cost 1.547528, exp_cost 1.595400, grad norm 10.219351, param norm 1478.210449, tps 935.396804, length mean/std 14.000000/0.000000
INFO:root:epoch 48, iter 6600, cost 1.577920, exp_cost 1.604806, grad norm 12.763432, param norm 1478.539185, tps 935.555276, length mean/std 20.000000/0.000000
INFO:root:epoch 48, iter 6800, cost 1.654132, exp_cost 1.592098, grad norm 7.450541, param norm 1478.858398, tps 935.702389, length mean/std 10.000000/0.000000
INFO:root:epoch 48, iter 7000, cost 1.949255, exp_cost 1.607722, grad norm 3.181066, param norm 1479.136108, tps 935.884252, length mean/std 3.000000/0.000000
INFO:root:epoch 48, iter 7200, cost 1.708109, exp_cost 1.608974, grad norm 15.853533, param norm 1479.363647, tps 936.042789, length mean/std 23.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 258697325 get requests, put_count=258697327 evicted_count=263000 eviction_rate=0.00101663 and unsatisfied allocation rate=0.00101849
INFO:root:epoch 48, iter 7400, cost 1.519670, exp_cost 1.603146, grad norm 13.975231, param norm 1479.639648, tps 936.212013, length mean/std 22.000000/0.000000
INFO:root:epoch 48, iter 7600, cost 1.617536, exp_cost 1.601519, grad norm 7.858808, param norm 1479.952393, tps 936.373042, length mean/std 11.000000/0.000000
INFO:root:epoch 48, iter 7800, cost 1.638928, exp_cost 1.599247, grad norm 19.104315, param norm 1480.182617, tps 936.538539, length mean/std 30.000000/0.000000
INFO:root:epoch 48, iter 8000, cost 1.573268, exp_cost 1.606360, grad norm 12.378471, param norm 1480.497681, tps 936.704044, length mean/std 17.000000/0.000000
INFO:root:epoch 48, iter 8200, cost 1.536054, exp_cost 1.601291, grad norm 9.868631, param norm 1480.781494, tps 936.871379, length mean/std 15.000000/0.000000
INFO:root:epoch 48, iter 8400, cost 1.559832, exp_cost 1.599967, grad norm 12.796465, param norm 1481.074219, tps 937.015716, length mean/std 20.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 263571346 get requests, put_count=263571024 evicted_count=273000 eviction_rate=0.00103577 and unsatisfied allocation rate=0.00103882
INFO:root:epoch 48, iter 8600, cost 1.479956, exp_cost 1.600540, grad norm 13.775773, param norm 1481.386963, tps 937.167067, length mean/std 23.000000/0.000000
INFO:root:epoch 48, iter 8800, cost 1.644278, exp_cost 1.595996, grad norm 3.824116, param norm 1481.664917, tps 937.309325, length mean/std 5.000000/0.000000
INFO:root:epoch 48, iter 9000, cost 1.624387, exp_cost 1.605327, grad norm 21.633282, param norm 1481.969604, tps 937.465373, length mean/std 36.000000/0.000000
INFO:root:epoch 48, iter 9200, cost 1.635793, exp_cost 1.606051, grad norm 22.550552, param norm 1482.234375, tps 937.624430, length mean/std 37.000000/0.000000
INFO:root:epoch 48, iter 9400, cost 1.535939, exp_cost 1.599118, grad norm 15.469253, param norm 1482.511963, tps 937.786332, length mean/std 25.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 268195044 get requests, put_count=268194726 evicted_count=283000 eviction_rate=0.0010552 and unsatisfied allocation rate=0.00105819
INFO:root:epoch 48, iter 9600, cost 1.476113, exp_cost 1.594679, grad norm 13.780165, param norm 1482.713989, tps 937.948016, length mean/std 23.000000/0.000000
INFO:root:epoch 48, iter 9800, cost 2.018140, exp_cost 1.598040, grad norm 2.695145, param norm 1482.980591, tps 938.106691, length mean/std 3.000000/0.000000
INFO:root:epoch 48, iter 10000, cost 1.456541, exp_cost 1.595785, grad norm 11.527656, param norm 1483.309570, tps 938.260763, length mean/std 17.000000/0.000000
INFO:root:epoch 48, iter 10200, cost 1.896613, exp_cost 1.605338, grad norm 3.348327, param norm 1483.633179, tps 938.413318, length mean/std 4.000000/0.000000
INFO:root:epoch 48, iter 10400, cost 1.694608, exp_cost 1.602062, grad norm 16.594549, param norm 1483.932617, tps 938.554169, length mean/std 24.000000/0.000000
INFO:root:epoch 48, iter 10600, cost 1.549076, exp_cost 1.600860, grad norm 13.171162, param norm 1484.295166, tps 938.704792, length mean/std 21.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 273440735 get requests, put_count=273440698 evicted_count=293000 eviction_rate=0.00107153 and unsatisfied allocation rate=0.00107343
INFO:root:epoch 48, iter 10800, cost 1.568551, exp_cost 1.611628, grad norm 11.809040, param norm 1484.548950, tps 938.865473, length mean/std 18.000000/0.000000
INFO:root:epoch 48, iter 11000, cost 1.570338, exp_cost 1.609680, grad norm 12.307176, param norm 1484.801270, tps 939.010949, length mean/std 19.000000/0.000000
INFO:root:epoch 48, iter 11200, cost 1.831732, exp_cost 1.607474, grad norm 2.715347, param norm 1485.119385, tps 939.170456, length mean/std 3.000000/0.000000
INFO:root:epoch 48, iter 11400, cost 1.473273, exp_cost 1.601878, grad norm 7.995485, param norm 1485.391846, tps 939.307249, length mean/std 12.000000/0.000000
INFO:root:epoch 48, iter 11600, cost 2.264940, exp_cost 1.601417, grad norm 1.889514, param norm 1485.619019, tps 939.429267, length mean/std 2.000000/0.000000
INFO:root:epoch 48, iter 11800, cost 1.520776, exp_cost 1.606536, grad norm 13.310073, param norm 1485.900269, tps 939.578598, length mean/std 20.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 278076428 get requests, put_count=278076378 evicted_count=303000 eviction_rate=0.00108963 and unsatisfied allocation rate=0.00109154
INFO:root:epoch 48, iter 12000, cost 1.603696, exp_cost 1.597683, grad norm 5.289315, param norm 1486.147583, tps 939.709570, length mean/std 7.000000/0.000000
INFO:root:epoch 48, iter 12200, cost 1.685061, exp_cost 1.599188, grad norm 21.440676, param norm 1486.393188, tps 939.850607, length mean/std 34.000000/0.000000
INFO:root:epoch 48, iter 12400, cost 1.650500, exp_cost 1.605140, grad norm 5.912076, param norm 1486.657959, tps 939.998992, length mean/std 8.000000/0.000000
INFO:root:Epoch 48 Validation cost: 1.699129 time: 7283.649051
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-48 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-48 is not in all_model_checkpoint_paths. Manually adding it.
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 282847022 get requests, put_count=282846905 evicted_count=313000 eviction_rate=0.00110661 and unsatisfied allocation rate=0.00110872
INFO:root:epoch 49, iter 200, cost 1.542799, exp_cost 1.585851, grad norm 7.026083, param norm 1487.072144, tps 931.073158, length mean/std 10.000000/0.000000
INFO:root:epoch 49, iter 400, cost 1.627228, exp_cost 1.574413, grad norm 20.868771, param norm 1487.346191, tps 931.199215, length mean/std 34.000000/0.000000
INFO:root:epoch 49, iter 600, cost 1.487398, exp_cost 1.590074, grad norm 9.192636, param norm 1487.641602, tps 931.361811, length mean/std 13.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 285537203 get requests, put_count=285537179 evicted_count=323000 eviction_rate=0.0011312 and unsatisfied allocation rate=0.00113297
INFO:root:epoch 49, iter 800, cost 1.559912, exp_cost 1.581655, grad norm 16.386600, param norm 1487.929932, tps 931.554374, length mean/std 26.000000/0.000000
INFO:root:epoch 49, iter 1000, cost 1.616029, exp_cost 1.580421, grad norm 17.926439, param norm 1488.192993, tps 931.697427, length mean/std 29.000000/0.000000
INFO:root:epoch 49, iter 1200, cost 1.667257, exp_cost 1.587255, grad norm 23.856869, param norm 1488.482178, tps 931.853554, length mean/std 36.000000/0.000000
INFO:root:epoch 49, iter 1400, cost 1.620366, exp_cost 1.594628, grad norm 15.830956, param norm 1488.740234, tps 932.005497, length mean/std 26.000000/0.000000
INFO:root:epoch 49, iter 1600, cost 1.653110, exp_cost 1.579680, grad norm 20.668148, param norm 1489.017090, tps 932.166598, length mean/std 35.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 289667732 get requests, put_count=289667651 evicted_count=333000 eviction_rate=0.00114959 and unsatisfied allocation rate=0.00115154
INFO:root:epoch 49, iter 1800, cost 1.642251, exp_cost 1.579261, grad norm 21.580814, param norm 1489.255127, tps 932.328405, length mean/std 35.000000/0.000000
INFO:root:epoch 49, iter 2000, cost 1.603099, exp_cost 1.595533, grad norm 8.342435, param norm 1489.485474, tps 932.494496, length mean/std 11.000000/0.000000
INFO:root:epoch 49, iter 2200, cost 1.491743, exp_cost 1.597991, grad norm 9.906278, param norm 1489.824097, tps 932.647400, length mean/std 15.000000/0.000000
INFO:root:epoch 49, iter 2400, cost 1.547117, exp_cost 1.578009, grad norm 17.571070, param norm 1490.076172, tps 932.786653, length mean/std 28.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 293455064 get requests, put_count=293455012 evicted_count=343000 eviction_rate=0.00116883 and unsatisfied allocation rate=0.00117065
INFO:root:epoch 49, iter 2600, cost 1.562161, exp_cost 1.591607, grad norm 13.486195, param norm 1490.415161, tps 932.937364, length mean/std 20.000000/0.000000
INFO:root:epoch 49, iter 2800, cost 1.907617, exp_cost 1.591485, grad norm 3.432805, param norm 1490.704468, tps 933.090970, length mean/std 4.000000/0.000000
INFO:root:epoch 49, iter 3000, cost 1.538718, exp_cost 1.596652, grad norm 8.768224, param norm 1490.946655, tps 933.249896, length mean/std 13.000000/0.000000
INFO:root:epoch 49, iter 3200, cost 1.578316, exp_cost 1.589918, grad norm 10.969277, param norm 1491.189697, tps 933.409517, length mean/std 17.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 296586128 get requests, put_count=296586122 evicted_count=353000 eviction_rate=0.00119021 and unsatisfied allocation rate=0.00119186
INFO:root:epoch 49, iter 3400, cost 1.598338, exp_cost 1.591623, grad norm 8.759449, param norm 1491.381348, tps 933.549635, length mean/std 12.000000/0.000000
INFO:root:epoch 49, iter 3600, cost 1.541902, exp_cost 1.586598, grad norm 5.997016, param norm 1491.673706, tps 933.699830, length mean/std 9.000000/0.000000
INFO:root:epoch 49, iter 3800, cost 1.689446, exp_cost 1.594588, grad norm 27.277473, param norm 1491.933350, tps 933.839406, length mean/std 44.000000/0.000000
INFO:root:epoch 49, iter 4000, cost 1.495556, exp_cost 1.592629, grad norm 11.082550, param norm 1492.207764, tps 933.983486, length mean/std 17.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 300309523 get requests, put_count=300309476 evicted_count=363000 eviction_rate=0.00120875 and unsatisfied allocation rate=0.00121051
INFO:root:epoch 49, iter 4200, cost 1.886346, exp_cost 1.584946, grad norm 6.084820, param norm 1492.474243, tps 934.128356, length mean/std 7.000000/0.000000
INFO:root:epoch 49, iter 4400, cost 2.266165, exp_cost 1.590396, grad norm 2.059989, param norm 1492.811157, tps 934.271018, length mean/std 2.000000/0.000000
INFO:root:epoch 49, iter 4600, cost 1.469951, exp_cost 1.577182, grad norm 9.538382, param norm 1493.057251, tps 934.422988, length mean/std 15.000000/0.000000
INFO:root:epoch 49, iter 4800, cost 1.502216, exp_cost 1.585220, grad norm 9.597644, param norm 1493.315918, tps 934.560809, length mean/std 14.000000/0.000000
INFO:root:epoch 49, iter 5000, cost 1.838194, exp_cost 1.586379, grad norm 2.693027, param norm 1493.641968, tps 934.703507, length mean/std 3.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 304156941 get requests, put_count=304156847 evicted_count=373000 eviction_rate=0.00122634 and unsatisfied allocation rate=0.00122823
INFO:root:epoch 49, iter 5200, cost 1.530140, exp_cost 1.592808, grad norm 7.950277, param norm 1493.985229, tps 934.857557, length mean/std 11.000000/0.000000
INFO:root:epoch 49, iter 5400, cost 2.999279, exp_cost 1.589832, grad norm 0.786455, param norm 1494.239136, tps 935.016005, length mean/std 1.000000/0.000000
INFO:root:epoch 49, iter 5600, cost 1.628276, exp_cost 1.591730, grad norm 21.242437, param norm 1494.465698, tps 935.164109, length mean/std 35.000000/0.000000
INFO:root:epoch 49, iter 5800, cost 1.537434, exp_cost 1.587764, grad norm 14.548196, param norm 1494.714233, tps 935.327431, length mean/std 23.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 307627991 get requests, put_count=307627867 evicted_count=383000 eviction_rate=0.00124501 and unsatisfied allocation rate=0.00124698
INFO:root:epoch 49, iter 6000, cost 1.652251, exp_cost 1.592391, grad norm 20.788767, param norm 1494.972046, tps 935.481647, length mean/std 31.000000/0.000000
INFO:root:epoch 49, iter 6200, cost 1.525116, exp_cost 1.584422, grad norm 12.376474, param norm 1495.235107, tps 935.625678, length mean/std 20.000000/0.000000
INFO:root:epoch 49, iter 6400, cost 1.699066, exp_cost 1.591203, grad norm 25.778435, param norm 1495.635010, tps 935.779996, length mean/std 38.000000/0.000000
INFO:root:epoch 49, iter 6600, cost 1.602799, exp_cost 1.591659, grad norm 16.986853, param norm 1495.879761, tps 935.929313, length mean/std 27.000000/0.000000
INFO:root:epoch 49, iter 6800, cost 1.552649, exp_cost 1.588781, grad norm 11.518511, param norm 1496.186890, tps 936.076234, length mean/std 18.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 311874188 get requests, put_count=311874036 evicted_count=393000 eviction_rate=0.00126012 and unsatisfied allocation rate=0.00126216
INFO:root:epoch 49, iter 7000, cost 1.476797, exp_cost 1.585988, grad norm 6.923863, param norm 1496.445557, tps 936.223085, length mean/std 10.000000/0.000000
INFO:root:epoch 49, iter 7200, cost 1.430928, exp_cost 1.582725, grad norm 6.722665, param norm 1496.781616, tps 936.368326, length mean/std 10.000000/0.000000
INFO:root:epoch 49, iter 7400, cost 1.583244, exp_cost 1.594430, grad norm 7.206411, param norm 1497.067749, tps 936.505379, length mean/std 10.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 315265687 get requests, put_count=315265289 evicted_count=403000 eviction_rate=0.00127829 and unsatisfied allocation rate=0.00128108
INFO:root:epoch 49, iter 7600, cost 1.556304, exp_cost 1.605077, grad norm 9.892159, param norm 1497.346680, tps 936.634630, length mean/std 14.000000/0.000000
INFO:root:epoch 49, iter 7800, cost 1.444434, exp_cost 1.600917, grad norm 8.986524, param norm 1497.641479, tps 936.780490, length mean/std 14.000000/0.000000
INFO:root:epoch 49, iter 8000, cost 1.949462, exp_cost 1.593975, grad norm 3.820310, param norm 1497.875732, tps 936.918774, length mean/std 4.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 317861263 get requests, put_count=317861211 evicted_count=413000 eviction_rate=0.00129931 and unsatisfied allocation rate=0.00130099
INFO:root:epoch 49, iter 8200, cost 1.521017, exp_cost 1.596662, grad norm 17.237364, param norm 1498.098755, tps 937.068720, length mean/std 27.000000/0.000000
INFO:root:epoch 49, iter 8400, cost 1.583372, exp_cost 1.600112, grad norm 16.329128, param norm 1498.499390, tps 937.206445, length mean/std 24.000000/0.000000
INFO:root:epoch 49, iter 8600, cost 1.539676, exp_cost 1.593413, grad norm 6.718211, param norm 1498.681763, tps 937.342666, length mean/std 9.000000/0.000000
INFO:root:epoch 49, iter 8800, cost 1.564923, exp_cost 1.598623, grad norm 17.605755, param norm 1498.981934, tps 937.475980, length mean/std 30.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 321255138 get requests, put_count=321254965 evicted_count=423000 eviction_rate=0.00131671 and unsatisfied allocation rate=0.00131875
INFO:root:epoch 49, iter 9000, cost 1.555788, exp_cost 1.581444, grad norm 5.146841, param norm 1499.233643, tps 937.626909, length mean/std 7.000000/0.000000
INFO:root:epoch 49, iter 9200, cost 1.498696, exp_cost 1.597425, grad norm 11.857836, param norm 1499.546265, tps 937.763092, length mean/std 18.000000/0.000000
INFO:root:epoch 49, iter 9400, cost 1.495364, exp_cost 1.607911, grad norm 14.074768, param norm 1499.850464, tps 937.928236, length mean/std 20.000000/0.000000
INFO:root:epoch 49, iter 9600, cost 1.639366, exp_cost 1.599574, grad norm 23.544016, param norm 1500.094971, tps 938.063727, length mean/std 38.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 324628132 get requests, put_count=324627842 evicted_count=433000 eviction_rate=0.00133384 and unsatisfied allocation rate=0.00133621
INFO:root:epoch 49, iter 9800, cost 1.637926, exp_cost 1.592287, grad norm 7.621647, param norm 1500.343750, tps 938.212854, length mean/std 10.000000/0.000000
INFO:root:epoch 49, iter 10000, cost 1.639813, exp_cost 1.598440, grad norm 5.272971, param norm 1500.623291, tps 938.358927, length mean/std 7.000000/0.000000
INFO:root:epoch 49, iter 10200, cost 1.552087, exp_cost 1.602217, grad norm 15.506730, param norm 1500.873535, tps 938.481551, length mean/std 23.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 327221705 get requests, put_count=327221230 evicted_count=443000 eviction_rate=0.00135382 and unsatisfied allocation rate=0.00135675
INFO:root:epoch 49, iter 10400, cost 1.629881, exp_cost 1.599333, grad norm 13.102344, param norm 1501.103394, tps 938.621217, length mean/std 19.000000/0.000000
INFO:root:epoch 49, iter 10600, cost 1.562628, exp_cost 1.602146, grad norm 7.248018, param norm 1501.431152, tps 938.782160, length mean/std 10.000000/0.000000
INFO:root:epoch 49, iter 10800, cost 1.702073, exp_cost 1.611296, grad norm 23.392010, param norm 1501.703491, tps 938.907798, length mean/std 37.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 330044264 get requests, put_count=330043881 evicted_count=453000 eviction_rate=0.00137254 and unsatisfied allocation rate=0.00137516
INFO:root:epoch 49, iter 11000, cost 1.664970, exp_cost 1.610590, grad norm 20.390476, param norm 1501.989624, tps 939.048001, length mean/std 33.000000/0.000000
INFO:root:epoch 49, iter 11200, cost 1.663031, exp_cost 1.606851, grad norm 24.237616, param norm 1502.223022, tps 939.187549, length mean/std 39.000000/0.000000
INFO:root:epoch 49, iter 11400, cost 1.535175, exp_cost 1.604000, grad norm 7.377937, param norm 1502.495728, tps 939.311244, length mean/std 10.000000/0.000000
INFO:root:epoch 49, iter 11600, cost 1.482979, exp_cost 1.597445, grad norm 9.695393, param norm 1502.886963, tps 939.441992, length mean/std 14.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 333412031 get requests, put_count=333412008 evicted_count=463000 eviction_rate=0.00138867 and unsatisfied allocation rate=0.00139019
INFO:root:epoch 49, iter 11800, cost 1.714846, exp_cost 1.605720, grad norm 7.018185, param norm 1503.145630, tps 939.588056, length mean/std 8.000000/0.000000
INFO:root:epoch 49, iter 12000, cost 1.633862, exp_cost 1.594891, grad norm 5.084870, param norm 1503.455078, tps 939.731542, length mean/std 7.000000/0.000000
INFO:root:epoch 49, iter 12200, cost 1.590666, exp_cost 1.598690, grad norm 13.845526, param norm 1503.755005, tps 939.873786, length mean/std 20.000000/0.000000
INFO:root:epoch 49, iter 12400, cost 1.421376, exp_cost 1.598280, grad norm 5.712060, param norm 1504.057739, tps 940.004440, length mean/std 9.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 336701161 get requests, put_count=336701095 evicted_count=473000 eviction_rate=0.00140481 and unsatisfied allocation rate=0.00140643
INFO:root:Epoch 49 Validation cost: 1.698023 time: 7363.650058
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-49 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:/gss_gpfs_scratch/dong.r/Model/BCI/Sublex/1/1/new/s2s_sub_random_fix_49_5_bi/best.ckpt-49 is not in all_model_checkpoint_paths. Manually adding it.
INFO:root:epoch 50, iter 200, cost 1.557048, exp_cost 1.584047, grad norm 20.370951, param norm 1504.420898, tps 931.871794, length mean/std 30.000000/0.000000
INFO:root:epoch 50, iter 400, cost 1.641800, exp_cost 1.580639, grad norm 21.443218, param norm 1504.678467, tps 932.038629, length mean/std 34.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 340484835 get requests, put_count=340484702 evicted_count=483000 eviction_rate=0.00141857 and unsatisfied allocation rate=0.00142037
INFO:root:epoch 50, iter 600, cost 1.547488, exp_cost 1.573721, grad norm 15.126620, param norm 1505.028442, tps 932.188322, length mean/std 22.000000/0.000000
INFO:root:epoch 50, iter 800, cost 2.996105, exp_cost 1.573310, grad norm 0.732515, param norm 1505.323242, tps 932.334949, length mean/std 1.000000/0.000000
INFO:root:epoch 50, iter 1000, cost 1.913891, exp_cost 1.579412, grad norm 3.550734, param norm 1505.590942, tps 932.496613, length mean/std 4.000000/0.000000
INFO:root:epoch 50, iter 1200, cost 1.580716, exp_cost 1.591031, grad norm 24.746647, param norm 1505.883301, tps 932.658445, length mean/std 42.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 344080980 get requests, put_count=344080806 evicted_count=493000 eviction_rate=0.0014328 and unsatisfied allocation rate=0.00143471
INFO:root:epoch 50, iter 1400, cost 1.590485, exp_cost 1.581051, grad norm 23.450546, param norm 1506.140991, tps 932.816905, length mean/std 38.000000/0.000000
INFO:root:epoch 50, iter 1600, cost 1.989618, exp_cost 1.585504, grad norm 2.653893, param norm 1506.457642, tps 932.954600, length mean/std 3.000000/0.000000
INFO:root:epoch 50, iter 1800, cost 1.545043, exp_cost 1.572155, grad norm 21.190449, param norm 1506.757446, tps 933.101460, length mean/std 34.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 346799295 get requests, put_count=346799007 evicted_count=503000 eviction_rate=0.00145041 and unsatisfied allocation rate=0.00145263
INFO:root:epoch 50, iter 2000, cost 1.555118, exp_cost 1.582913, grad norm 22.738359, param norm 1506.983276, tps 933.247621, length mean/std 39.000000/0.000000
INFO:root:epoch 50, iter 2200, cost 2.945552, exp_cost 1.587912, grad norm 0.753877, param norm 1507.295288, tps 933.392604, length mean/std 1.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 348798564 get requests, put_count=348798568 evicted_count=513000 eviction_rate=0.00147076 and unsatisfied allocation rate=0.00147213
INFO:root:epoch 50, iter 2400, cost 1.585560, exp_cost 1.592915, grad norm 3.108050, param norm 1507.512817, tps 933.537217, length mean/std 4.000000/0.000000
INFO:root:epoch 50, iter 2600, cost 1.698639, exp_cost 1.582632, grad norm 5.450014, param norm 1507.847412, tps 933.660274, length mean/std 7.000000/0.000000
INFO:root:epoch 50, iter 2800, cost 1.476189, exp_cost 1.601502, grad norm 9.402134, param norm 1508.016968, tps 933.804903, length mean/std 14.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 350886235 get requests, put_count=350886192 evicted_count=523000 eviction_rate=0.00149051 and unsatisfied allocation rate=0.00149201
INFO:root:epoch 50, iter 3000, cost 1.531103, exp_cost 1.586418, grad norm 14.253352, param norm 1508.325317, tps 933.945681, length mean/std 21.000000/0.000000
INFO:root:epoch 50, iter 3200, cost 1.552572, exp_cost 1.585463, grad norm 18.341715, param norm 1508.642090, tps 934.080065, length mean/std 27.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 353157654 get requests, put_count=353157339 evicted_count=533000 eviction_rate=0.00150924 and unsatisfied allocation rate=0.0015115
INFO:root:epoch 50, iter 3400, cost 1.574219, exp_cost 1.595586, grad norm 7.122507, param norm 1508.979126, tps 934.224506, length mean/std 10.000000/0.000000
INFO:root:epoch 50, iter 3600, cost 1.495906, exp_cost 1.596859, grad norm 6.964383, param norm 1509.138794, tps 934.371604, length mean/std 11.000000/0.000000
INFO:root:epoch 50, iter 3800, cost 1.960846, exp_cost 1.597867, grad norm 4.431844, param norm 1509.375488, tps 934.498113, length mean/std 5.000000/0.000000
INFO:root:epoch 50, iter 4000, cost 1.558386, exp_cost 1.579181, grad norm 6.996970, param norm 1509.718018, tps 934.632333, length mean/std 9.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 356102637 get requests, put_count=356102220 evicted_count=543000 eviction_rate=0.00152484 and unsatisfied allocation rate=0.00152737
INFO:root:epoch 50, iter 4200, cost 1.493706, exp_cost 1.576903, grad norm 17.974197, param norm 1510.004272, tps 934.770935, length mean/std 31.000000/0.000000
INFO:root:epoch 50, iter 4400, cost 1.458817, exp_cost 1.591437, grad norm 9.830658, param norm 1510.312256, tps 934.899314, length mean/std 16.000000/0.000000
INFO:root:epoch 50, iter 4600, cost 1.705494, exp_cost 1.593833, grad norm 3.298962, param norm 1510.516357, tps 935.017884, length mean/std 4.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 359121601 get requests, put_count=359121601 evicted_count=553000 eviction_rate=0.00153987 and unsatisfied allocation rate=0.00154121
INFO:root:epoch 50, iter 4800, cost 1.600279, exp_cost 1.588004, grad norm 7.978975, param norm 1510.769165, tps 935.168260, length mean/std 11.000000/0.000000
INFO:root:epoch 50, iter 5000, cost 1.456368, exp_cost 1.587124, grad norm 14.826900, param norm 1511.079956, tps 935.313229, length mean/std 25.000000/0.000000
INFO:root:epoch 50, iter 5200, cost 1.542924, exp_cost 1.591831, grad norm 16.629877, param norm 1511.310059, tps 935.448144, length mean/std 28.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 362094273 get requests, put_count=362094097 evicted_count=563000 eviction_rate=0.00155484 and unsatisfied allocation rate=0.00155666
INFO:root:epoch 50, iter 5400, cost 1.605149, exp_cost 1.582931, grad norm 3.609509, param norm 1511.640259, tps 935.578295, length mean/std 4.000000/0.000000
INFO:root:epoch 50, iter 5600, cost 1.583322, exp_cost 1.585978, grad norm 5.303190, param norm 1511.857910, tps 935.718620, length mean/std 7.000000/0.000000
INFO:root:epoch 50, iter 5800, cost 1.673433, exp_cost 1.587656, grad norm 25.058784, param norm 1512.084839, tps 935.843277, length mean/std 37.000000/0.000000
INFO:root:epoch 50, iter 6000, cost 1.660481, exp_cost 1.593194, grad norm 22.394556, param norm 1512.405884, tps 935.976555, length mean/std 35.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 365052433 get requests, put_count=365052301 evicted_count=573000 eviction_rate=0.00156964 and unsatisfied allocation rate=0.00157132
INFO:root:epoch 50, iter 6200, cost 1.983016, exp_cost 1.582929, grad norm 3.640509, param norm 1512.595337, tps 936.119906, length mean/std 4.000000/0.000000
INFO:root:epoch 50, iter 6400, cost 1.827550, exp_cost 1.596507, grad norm 4.576913, param norm 1512.873535, tps 936.248948, length mean/std 5.000000/0.000000
INFO:root:epoch 50, iter 6600, cost 1.605856, exp_cost 1.601535, grad norm 14.261016, param norm 1513.100586, tps 936.397786, length mean/std 22.000000/0.000000
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 367712862 get requests, put_count=367712385 evicted_count=583000 eviction_rate=0.00158548 and unsatisfied allocation rate=0.00158808
slurmstepd: *** JOB 15720979 CANCELLED AT 2018-08-08T14:03:40 DUE TO TIME LIMIT on compute-2-136 ***
