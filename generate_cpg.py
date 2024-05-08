import os, sys
import shutil

path = 'dataset/positive/'
# path = './neg/'
os.system('chmod -R +x ./joern-cli')

for commit in os.listdir(path):
    if os.path.exists(path + commit + '/outA'):
        print(commit + ' has been processed')
        continue
    os.system('mkdir ' + path + commit + '/outA')
    os.system('mkdir ' + path + commit + '/outB')
    if not os.path.exists(path + commit + '/a'):
        os.system('mkdir ' + path + commit + '/a')
    if not os.path.exists(path + commit + '/b'):
        os.system('mkdir ' + path + commit + '/b')
    for file_a in os.listdir(path + commit + '/a'):
        if file_a in os.listdir(path + commit + '/b'):
            os.system('./joern-cli/joern-parse ' + path + commit + '/a/' + file_a)
            os.system('./joern-cli/joern --script libs/locate_func.sc --param outFile=' + path + commit + '/cpg_a_' + file_a[:-5] + '.txt')
            os.system('./joern-cli/joern-parse ' + path + commit + '/b/' + file_a)
            os.system('./joern-cli/joern --script libs/locate_func.sc --param outFile=' + path + commit + '/cpg_b_' + file_a[:-5] + '.txt')
            os.system('python3 libs/locate_and_align.py ' +path + commit + '/')
    for file_a in os.listdir(path + commit + '/a'):
        os.system('./joern-cli/joern-parse ' + path + commit + '/a/' + file_a)
        os.system('./joern-cli/joern-export --repr cpg14 --out ' + path + commit + '/outA/' + file_a[:-5])
    for file_b in os.listdir(path + commit + '/b'):
        os.system('./joern-cli/joern-parse ' + path + commit + '/b/' + file_b)
        os.system('./joern-cli/joern-export --repr cpg14 --out ' +path + commit + '/outB/' + file_b[:-5])



