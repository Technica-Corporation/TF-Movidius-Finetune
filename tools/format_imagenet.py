import glob
import argparse
import csv
import os
import progressbar
from shutil import copyfile

parser = argparse.ArgumentParser(description='')
parser.add_argument('--labels_file', type=str, required=True, help='Path to labels files')
parser.add_argument('--training_labels', type=str, required=True, help='Path to labels files')
parser.add_argument('--base_folder', type=str, required=True, help='Path to labels files')
parser.add_argument('--dest_folder', type=str, required=True, help='Path to labels files')
args=parser.parse_args()

def write_labels(dest_folder, categories, groups):
    file = open(dest_folder+'labels.txt', 'w')
    for category, values in groups.items():
        file.write('{}:{}\n'.format(category, categories[category]))
    file.close()

def move_files(file_groups, categories, base_folder, dest_folder):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    bar = progressbar.ProgressBar()
    for category, images in bar(file_groups.items()):
        for i in images:
            # Check if this class exists.
            if not os.path.exists(os.path.join(dest_folder, str(category))):
                print("Creating folder for %s/%s" % (dest_folder, str(category)))
                os.makedirs(os.path.join(dest_folder, str(category)))
            # Check if we have already moved this file, or at least that it exists to move.
            if not os.path.exists(os.path.join(base_folder, i)):
                print("Can't find %s to move. Skipping." % (os.path.join(base_folder, i)))
                continue
            dest = os.path.join(dest_folder, str(category), i)
            if not os.path.exists(dest):
                #print("Moving %s/%s to %s" % (base_folder, i, dest))
                copyfile(os.path.join(base_folder, i), dest)


cats = open(args.labels_file, 'r')
categories = {}
ind = 1
for line in cats:
    categories[ind] = line.rstrip()
    ind+=1

cat_count = {}
with open(args.training_labels, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for r in reader:
        ind = int(r[1])
        if ind in cat_count:
            cat_count[ind].append(r[0])
        else:
            cat_count[ind] = []
            cat_count[ind].append(r[0])
#for k,v in cat_count.items():
#   print('Category: {}, Len: {}'.format(categories[k], len(v)))
move_files(cat_count, categories, args.base_folder, args.dest_folder)
write_labels(args.dest_folder, categories, cat_count)