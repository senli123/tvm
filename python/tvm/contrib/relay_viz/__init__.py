# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Relay IR Visualizer"""
from typing import Dict
import tvm
from tvm import relay
from .interface import (
    Plotter,
    VizGraph,
    VizParser,
)
from .terminal import (
    TermPlotter,
    TermVizParser,
)
from .dot import (
    DotPlotter,
    DotVizParser,
)


class RelayVisualizer:
    """Relay IR Visualizer

    Parameters
    ----------
    relay_mod: tvm.IRModule
        Relay IR module.
    relay_param: None | Dict[str, tvm.runtime.NDArray]
        Relay parameter dictionary. Default `None`.
    plotter: Plotter
        An instance of class inheriting from Plotter interface.
        Default is an instance of `terminal.TermPlotter`.
    parser: VizParser
        An instance of class inheriting from VizParser interface.
        Default is an instance of `terminal.TermVizParser`.
    """

    def __init__(
        self,
        relay_mod: tvm.IRModule,
        relay_param: Dict[str, tvm.runtime.NDArray] = None,
        plotter: Plotter = None,
        parser: VizParser = None,
        visual_mod : int = 0,  #0: only for visual  1: only for parse 2: visual and parse
    ):
        if visual_mod == 0:
            
            self.visual_plotter = plotter if plotter is not None else TermPlotter()
            self.visual_parser = parser if parser is not None else TermVizParser()
            
        elif visual_mod == 1:
            
            self._plotter = TermPlotter()
            self._parser = TermVizParser()
           
        elif visual_mod == 2:
            
            self._plotter = TermPlotter()
            self._parser = TermVizParser()
            self.visual_plotter = plotter if plotter is not None else TermPlotter()
            self.visual_parser = parser if parser is not None else TermVizParser()
        else:
            assert False, 'visual mode only support 0, 1 or 2'
        self._visual_mod = visual_mod
        self._relay_param = relay_param if relay_param is not None else {}

        global_vars = relay_mod.get_global_vars()
        graph_names = []
        # If we have main function, put it to the first.
        # Then main function can be shown on the top.
        for gv_node in global_vars:
            if gv_node.name_hint == "main":
                graph_names.insert(0, gv_node.name_hint)
            else:
                graph_names.append(gv_node.name_hint)

        node_to_id = {}
        # callback to generate an unique string-ID for nodes.
        # node_count_offset ensure each node ID is still unique across subgraph.
        node_count_offset = 0

        def traverse_expr(node):
            if node in node_to_id:
                return
            node_to_id[node] = str(len(node_to_id) + node_count_offset)

        for name in graph_names:
            node_count_offset += len(node_to_id)
            node_to_id.clear()
            relay.analysis.post_order_visit(relay_mod[name], traverse_expr)
            if self._visual_mod == 0:
                visual_graph = self.visual_plotter.create_graph(name)
                self._add_nodes(visual_graph, node_to_id)
            elif self._visual_mod == 1:
                graph = self._plotter.create_graph(name)
                self._add_nodes(graph, node_to_id)
            elif self._visual_mod == 2:
                visual_graph = self.visual_plotter.create_graph(name)
                graph = self._plotter.create_graph(name)
                self._add_nodes(visual_graph, node_to_id)
                self._add_nodes(graph, node_to_id)
    
            print(123)

    def _add_nodes(self, graph: VizGraph, node_to_id: Dict[relay.Expr, str]):
        """add nodes and to the graph.

        Parameters
        ----------
        graph : VizGraph
            a VizGraph for nodes to be added to.

        node_to_id : Dict[relay.expr, str]
            a mapping from nodes to an unique ID.

        relay_param : Dict[str, tvm.runtime.NDarray]
            relay parameter dictionary.
        """
        for node in node_to_id:
            viz_node, viz_edges = self._parser.get_node_edges(node, self._relay_param, node_to_id)
            if viz_node is not None:
                graph.node(viz_node)
            for edge in viz_edges:
                graph.edge(edge)

    def render(self, filename: str = None) -> None:
        if self._visual_mod == 1:
            assert False, "if visual_node is 1, it is not support visual"
        self._plotter.render(filename=filename)
