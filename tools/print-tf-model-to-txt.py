
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import gfile
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef

sys.path.extend(["../","./"])
from util.tensor_io import save_variables,save_variables_1

try:
    from tensorflow_py_api import mmi,mpe
except ImportError:
    print("no mmi module")

try:
    from tf_chain_py_api import chainloss,chainxentloss
except ImportError:
    print("no chainloss module")

def _parse_input_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into GraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    return input_graph_def

def _parse_input_meta_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into MetaGraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input meta graph file '" + input_graph + "' does not exist!")
        return -1
    input_meta_graph_def = MetaGraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_meta_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_meta_graph_def)
    print("Loaded meta graph file '" + input_graph)
    return input_meta_graph_def



def main(input_graph='',input_checkpoint=''):

    if input_graph:
        input_graph_def = _parse_input_graph_proto(input_graph, False)
        for node in input_graph_def.node:
            node.device = ""
        _ = importer.import_graph_def(input_graph_def, name="")

    with session.Session() as sess:
        if input_meta_graph_def:
            restorer = saver_lib.import_meta_graph(
                    input_meta_graph_def, clear_devices=True)
            restorer.restore(sess, input_checkpoint)
            pass
        else:
            var_list = {}
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ":0")
                except KeyError:
                    # This tensor doesn't exist in the graph (for example it's
                    # 'global_step' or a similar housekeeping element) so skip it.
                    continue
                var_list[key] = tensor
        #var_list.pop('global_step')
            saver = saver_lib.Saver(
                    var_list=var_list)
            saver.restore(sess, input_checkpoint)
        
        #param = sess.run(var_list['LSTMlstmlayer3/multi_rnn_cell/cell_0/lstmlayer3/w_i_diag'])
        param = sess.run(var_list)
        save_variables_1(param, 'model.txt')

if __name__ == '__main__':

    input_meta_graph = sys.argv[1]
    input_checkpoint = sys.argv[2]
    output = sys.argv[3]
    input_meta_graph_def = _parse_input_meta_graph_proto(input_meta_graph, True)
    for node in input_meta_graph_def.graph_def.node:
        node.device = ""
    with session.Session() as sess:
        restorer = saver_lib.import_meta_graph(
                input_meta_graph_def, clear_devices=True)
        restorer.restore(sess, input_checkpoint)
        variables=trainable_variables()
        param = sess.run(variables)
        save_variables(variables, param, output)

