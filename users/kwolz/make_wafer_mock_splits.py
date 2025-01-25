# import numpy as np
import os
import shutil


def copytree(src, dst, symlinks=False, ignore=None):
    """
    Copy directory tree including files from one directory to another.
    """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


map_dir_old = "/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031"
new_dir = "/pscratch/sd/k/kwolz/share/e2e_2412/satp1_241031_er"

# Copy file tree frm map_dir_old to new_dir
# copytree(map_dir_old, new_dir)

# Rename all maps from
# "atomic_{ctime}_{wafer}_{freq_channel}_full_wmap{suffix}"
# to "atomic_{ctime}_{wafer}_{freq_channel}_{split_label}_wmap{suffix}",
# where split_detail="wafer_high" or "wafer_low" depending on whether
# wafer="ws0..3" or wafer="ws4..6".
for subdir1, dirs1, files1 in os.walk(new_dir):
    for dir1 in dirs1:
        # print(os.path.join(subdir, dir))
        for subdir2, dirs2, files2 in os.walk(os.path.join(subdir1, dir1)):
            for file2 in files2:
                f2 = os.path.join(new_dir, dir1, file2)
                if ("ws0" in file2 or "ws1" in file2 or "ws2" in file2 or "ws3" in file2):  # noqa
                    os.rename(f2, f2.replace("full", "wafer_low"))
                    print("wafer_low", f2)
                if "ws4" in file2 or "ws5" in file2 or "ws6" in file2:
                    os.rename(f2, f2.replace("full", "wafer_high"))
                    print("wafer_high", f2)
