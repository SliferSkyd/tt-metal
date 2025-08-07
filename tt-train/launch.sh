#!/bin/bash

# Reset the cards on this node
tt-smi -r

# Each node in the cluster has this information populated on
# what it's role as per the underlying topology is
role=$(cat /etc/exabox_role)

# Use that info to determine which process to run on the node
case $role in
    compute)
        "${BIN_DIR}/nano_gpt" -c "${CONFIG}"
        ;;
    aggregator)
        ${BIN_DIR}/nano_gpt_aggregator -c "${CONFIG}"
        ;;
    optimizer)
        ${BIN_DIR}/nano_gpt_optimizer -c "${CONFIG}"
        ;;
    *)
        ;;
esac