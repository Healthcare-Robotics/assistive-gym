# train in parallel and wait for all to complete
for i in {10..19}
do
    (SMPL_FILE="examples/data/slp3d/p001/s${i}.pkl"
    echo "$SMPL_FILE"
    python3 -m assistive_gym.train --env "HumanComfort-v1" --smpl-file "${SMPL_FILE}" --person-id "p001" --save-dir "trained_models" --train --robot-ik --handover-obj all) &
done
wait
