CUDA_VISIBLE_DEVICES= /usr/bin/python worker.py --log-dir cartpole --env-id CartPole-v0 --num-workers 4 --job-name ps

docker run -i --rm --net=host ardrone /usr/bin/python /catkin/src/a3c/worker.py --log-dir cartpole --env-id CartPole-v0 --num-workers 1 --job-name worker --job-name worker --task 0 --remotes 1
