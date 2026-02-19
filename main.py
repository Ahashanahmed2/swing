#github
#git remote add origin https://github.com/Ahashanahmed2/swing.git



#main.py
import sys
import subprocess
import os

scripts = [
    "scripts/mongodb.py",
    
    "scripts/generate_pdf.py",
    "scripts/dayliMassage.py",
    "scripts/email_reports.py",
    
]

for script in scripts:
    script_path=os.path.abspath(script)
    try:
        subprocess.run([sys.executable,script_path],check=True)
        print(f"Finished {script}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}:{e}")
        break
    except FileNotFoundError:
        print(f"File not found :{script}")
        break
