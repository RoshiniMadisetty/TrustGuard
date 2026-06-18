import pandas as pd
import random

# Load UCI firewall dataset
uci = pd.read_csv("uci_dataset.csv")

NUM_RECORDS = 10000

records = []

for i in range(NUM_RECORDS):

    row = uci.sample(1).iloc[0]

    dst_port = str(row["Destination Port"])
    action = str(row["Action"]).upper()

    requirement = (
        f"Allow traffic to destination port {dst_port}"
        if action == "ALLOW"
        else f"Block traffic to destination port {dst_port}"
    )

    label_type = random.choices(
        ["correct", "hallucinated", "dangerous"],
        weights=[60, 25, 15]
    )[0]

    generated_action = action
    hallucination_type = "none"
    reason = "Rule matches intent"

    if label_type == "hallucinated":
        generated_action = action

        wrong_port = str(random.randint(1, 65535))

        while wrong_port == dst_port:
            wrong_port = str(random.randint(1, 65535))

        dst_port_used = wrong_port

        hallucination_type = "wrong_port"
        reason = f"Wrong port generated ({wrong_port})"

    elif label_type == "dangerous":

        generated_action = (
            "ALLOW"
            if action != "ALLOW"
            else "ALLOW"
        )

        dst_port_used = dst_port

        hallucination_type = "over_permissive"
        reason = "Potential security violation"

    else:

        dst_port_used = dst_port

    records.append({
        "pair_id": f"UCI-{i+1:05d}",
        "requirement": requirement,
        "action": generated_action.lower(),
        "protocol": "tcp",
        "source": "any",
        "destination": "any",
        "source_port": "any",
        "destination_port": dst_port_used,
        "direction": "both",
        "label": label_type,
        "hallucination_type": hallucination_type,
        "label_confidence": 0.9,
        "reasons": reason
    })

df = pd.DataFrame(records)

df.to_csv(
    "trustguard_10000.csv",
    index=False
)

print(f"Generated {len(df)} records")