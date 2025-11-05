import logging
import numpy as np
import loggingTestDir.logginTestAux
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="test.log", encoding="utf-8", filemode="w", level=logging.DEBUG
)


logger.info("Something from loggingTest.py")

arr = np.array((3, 5))

logger.info(f"{arr}")

loggingTestDir.logginTestAux.main()

process = subprocess.run(
    ["bash", "testBash"],
    check=True,
    cwd="/Users/benmoore/projects/sandTrapShapeOptBenchmarking/qNParEgo",
)

for line in process.stdout:
    logger.info(line.strip())
# process.wait()
