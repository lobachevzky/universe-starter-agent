#!/usr/bin/env python
import argparse
import json
import logging
import rospy

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description=None)
parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
parser.add_argument('--task', default=0, type=int, help='Task index')
parser.add_argument('--job-name', default='worker', help='worker or ps')
parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
parser.add_argument('--log-dir', default='/tmp/pong', help='Log directory path')
parser.add_argument('--env-id', default='PongDeterministic-v3', help='Environment id')
parser.add_argument('--spec', type=str, default=None,
                    help="Path to file with spec (argument to tf.train.ClusterSpec)")
parser.add_argument('--host', default='127.0.0.1'
                    , help='ip address for parameter sever (docker0 if gazebo)')
parser.add_argument('-r', '--remotes', default=None,
                    help='References to environments to create (e.g. -r 20), '
                         'or the address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

# Add visualisation argument
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")

(args, _) = parser.parse_known_args()
print(args.spec)
spec = {}
for row in args.spec.split(';'):
    row = row.split(',')
    spec[row[0]] = row[1:]
print('SPEC')
print(spec)
# logger.info(json.loads(args.spec))

while not rospy.is_shutdown():
    pass
