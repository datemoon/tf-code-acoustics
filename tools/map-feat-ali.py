import sys

if len(sys.argv) != 4:
    print(sys.argv[0] + ' feat_scp sort_ali feat_outscp')
    sys.exit(1)

feat_scp=sys.argv[1]
ali=sys.argv[2]
feat_out=sys.argv[3]

output_featscp=open(feat_out,'w')

feat_dict={}
for line in open(feat_scp):
    val = line.strip()
    key = val.split()[0]
    feat_dict[key] = val

# ali is sort
for line in open(ali):
    key = line.strip().split()[0]
    try:
        feat_val = feat_dict[key]
        output_featscp.write(feat_val + '\n')
    except KeyError:
        print('no this feature '+ key +'\n')


output_featscp.close()

