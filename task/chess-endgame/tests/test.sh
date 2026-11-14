#!/bin/bash
# The score is computed by the game server, never by the agent's container: fetch it and
# write the reward. Any failure to reach the server scores 0.
mkdir -p /logs/verifier
python3 /tests/score.py || echo 0 > /logs/verifier/reward.txt
