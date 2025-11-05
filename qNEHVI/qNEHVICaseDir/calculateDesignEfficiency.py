import argparse
import numpy as np
import subprocess
import os


totalParticles = 7000
totalMass = 0.000444153  # Total mass in kg

escapedParticles = float(
    subprocess.check_output(
        "tac log.KPF | grep 'escape' | awk -F'=' '{print $2}' | sed -n 1p | cut -d',' -f1 | tr -d ' ()'",
        shell=True,
        text=True,
    ).strip()
)

escapedMass = float(
    subprocess.check_output(
        " tac log.KPF | grep 'escape' | awk -F'=' '{print $2}' | sed -n 1p | cut -d',' -f2 | tr -d ' ()'",
        shell=True,
        text=True,
    ).strip()
)

print(f"Escaped Particles: {(escapedParticles)}")
print(f"Escaped Mass: {(escapedMass)}")
numberEfficiency = (1 - (escapedParticles / totalParticles)) * 100
massEfficiency = (
    1 - (escapedMass / totalMass)
) * 100  # Assuming total mass is 0.000444153 kg

print(f"Number Efficiency: {numberEfficiency}")
print(f"Mass Efficiency: {massEfficiency}")

np.savetxt(
    "efficiency.txt",
    np.array([numberEfficiency, massEfficiency]),
    header="Number Efficiency, Mass Efficiency",
    delimiter=",",
    fmt="%f",
)
