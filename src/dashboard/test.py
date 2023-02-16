import glob
import os
print('curr '+str(os.listdir(os.curdir)))
csv_files = glob.glob("*")
print(csv_files)