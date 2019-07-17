
que=3gpuq

for i in `seq 0 10`; do
    srun -p $que ./test_time $config | tail -n 1 | cut -d ' ' -f 4 >> /tmp/test
done

awk '{s+=$1}END{print s/NR}' RS="\n" /tmp/test