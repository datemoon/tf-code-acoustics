import sys
import os


if len(sys.argv) != 5:
    print(sys.argv[0] + ' sort_feat_scp sort_ali split(1) outdir')
    sys.exit(1)

sort_feat_scp=sys.argv[1]
sort_ali=sys.argv[2]
sort_feat_scp
split=int(sys.argv[3])
outdir=sys.argv[4]

if not os.path.exists(outdir):
    os.makedirs(outdir)

feat_name=sort_feat_scp.split('/')[-1]
ali_name=sort_ali.split('/')[-1]

split_ali=[]
split_feat=[]
for i in range(0, split):
    split_ali.append(open(outdir+'/'+ali_name+str(i),'w'))
    split_feat.append(open(outdir+'/'+feat_name+str(i),'w'))

featnumline=0
for line in open(sort_feat_scp):
    split_feat[int(featnumline%split)].write(line)
    featnumline += 1

alinumline=0
for line in open(sort_ali):
    split_ali[int(alinumline%split)].write(line)
    alinumline += 1


assert featnumline == alinumline

for i in range(0, split):
    split_ali[i].close()
    split_feat[i].close()


