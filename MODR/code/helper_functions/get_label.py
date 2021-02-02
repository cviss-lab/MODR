import os
import numpy as np
import pandas as pd
from shutil import copyfile

pth = '../../datasets/V5/RIMG'
create_unique_dataset = True

# Find all files
all_imgs = []
for rt, dirs, files in os.walk(pth):
    if len(files) > 0:
        all_imgs.extend([[os.path.join(rt, f), rt.split('/')[-1]] for f in files])

if create_unique_dataset:
    for img in all_imgs:
        copyfile(img[0], 'unique_dataset/'+ img[0].split('/')[-1])

all_imgs = pd.DataFrame(all_imgs, columns=['full_path', 'folder'])



# Pivot dataframe
all_imgs['count'] = 1
all_imgs = all_imgs.pivot('full_path', 'folder','count')
all_imgs.reset_index(inplace=True)
all_imgs.fillna(0, inplace=True)
#
all_imgs['file'] = all_imgs['full_path'].str.split('/').str[-1]

# Create aggregation dictionary
test = {'full_path': 'first'}
for i in all_imgs.columns[1:-1]:
    test[i] = 'sum'
# Aggregate
all_imgs = all_imgs.groupby(['file'], as_index=False).agg(test)

# Stats
print("There are {} multilabelled pictures out of {} pictures".format(sum(all_imgs.sum(axis=1) > 1), len(all_imgs)))
print("There are a total of {} pictures".format(sum(all_imgs.sum(numeric_only=True))))

# Export to csv
all_imgs.to_csv('multilabels.csv')
