

def PosteriorToPdfMatrix(post, nnet_diff_h):
    '''
    post is [ {pdf1:post2,pdf2:post2,...}, ... ]
    '''
    for t in range(len(post)):
        for key in post.keys():
            nnet_diff_h[t][k] = post[key]


