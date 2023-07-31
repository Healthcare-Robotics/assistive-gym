# train in parallel and wait for all to complete
for i in {1..9}
do
    (SMPL_FILE="examples/data/smpl_bp_ros_smpl_${i}.pkl"
    echo "$SMPL_FILE"
    python3 -m assistive_gym.train --env "HumanComfort-v1" --smpl-file "${SMPL_FILE}" --save-dir "trained_models" --train --robot-ik --handover-obj all) &
done
wait
