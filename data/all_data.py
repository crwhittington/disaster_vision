import os
import subprocess
import sys

# install gdown if needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

# download the file
subprocess.check_call([
    sys.executable, "-m", "gdown",
    "--id", "1kMC2PCTyWoOiL0AItssA7Grh4CSoPO2K",
    "-O", "data.zip"
])

# make output directory
os.makedirs("data", exist_ok=True)

# unzip the file
subprocess.check_call(["unzip", "-q", "data.zip", "-d", "data"])

print("finished unzipping")