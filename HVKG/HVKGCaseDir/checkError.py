import argparse
import numpy as np
import subprocess
import os


maxCo = float(
    subprocess.check_output(
        "tac log.KPF | grep 'Courant' | awk -F'max:' '{print $2}' | sed -n 1p",
        shell=True,
        text=True,
    ).strip()
)
print("Max Co Number after initialisation:\n", maxCo)

np.savetxt("maxCo.txt", [maxCo])
