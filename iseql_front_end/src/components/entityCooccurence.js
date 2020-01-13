import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import Typography from '@material-ui/core/Typography';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import ListItemText from '@material-ui/core/ListItemText';
import withRoot from '../withRoot';
import GraphVis from '../visualizations/graph_vis';
import MatrixVis from '../visualizations/matrix_vis';
import EdgeBundleVis from '../visualizations/edge_bundle_vis';
import SortableTable from '../components/core/sortable_table';
import {is_valid} from '../utils';
import {construct_force_directed_data} from '../utils/graph_utils';

const styles = theme => ({
root: {
    textAlign: 'center',
    paddingTop: theme.spacing.unit * 20,
},
nested: {
    paddingLeft: theme.spacing.unit * 4,
},
fab: {
    margin: theme.spacing.unit,
},
visualization : {
    display: "inline-block",
    margin: theme.spacing.unit * 2
},
list : {
    height: theme.spacing.unit * 50,
    overflow: "scroll",
}
});

class EntityCoOcurrence extends React.Component {

    construct_co_occurence_graph(data) {
        if (!is_valid(data)) {
            return null;
        }
        const {show_predictions} = this.props;
        const id_entity = [];
        const entity_id = {};
        const graph = {}
        for (const s_id in data) {
            const entry_data = data[s_id];
            const sentence_data = entry_data[1];
            const entities = show_predictions ? sentence_data.entities : sentence_data.real_entities;
            for (const ei in entities) {
                const ei_ent = entities[ei];
                let ei_id = -1;
                if (!(ei_ent in entity_id)) {
                    ei_id = id_entity.length;
                    entity_id[ei_ent] = ei_id;
                    id_entity.push(ei_ent);
                } else {
                    ei_id = entity_id[ei_ent];
                }


                if (!(ei_id in graph)) {
                    graph[ei_id] = {targets : {}, count: 0};
                }
                graph[ei_id].count++;
                for (const eii in entities) {
                    const eii_ent = entities[eii];
                    let eii_id = -1;
                    if (!(eii_ent in entity_id)) {
                        eii_id = id_entity.length
                        entity_id[eii_ent] = eii_id;
                        id_entity.push(eii_ent);
                    } else {
                        eii_id = entity_id[eii_ent];
                    }
                    // connect ei and eii
                    if (!(eii_id in graph[ei_id].targets)) {
                        graph[ei_id].targets[eii_id] = 0;
                    }
                    graph[ei_id].targets[eii_id]++;
                }
            }
        }

        const new_graph = {}
        for (const source in graph) {
            if (Object.keys(graph[source].targets).length > 0) {
                new_graph[source] = graph[source];
            }
        }

        return {entity_id, id_entity, graph: new_graph};
    }

    construct_force_directed_data(graph_data=null) {
        graph_data = graph_data === null ? this.construct_co_occurence_graph(this.props.data) : graph_data;
        let entity_id, id_entity, graph = null;
        ({entity_id, id_entity, graph} = graph_data);
        const nodes = [];
        const edges = [];
        const value = 1;
        for (const source in graph) {
            const targets = graph[source].targets;
            const count = graph[source].count;
            nodes.push(
                {
                    name: id_entity[source],
                    group: 0,
                    index: source,
                    count,
                }
            );
            for (const target in targets) {
                const count = targets[target];
                if (source < target ){ 
                    const edge = {
                        source,
                        target,
                        value,
                        count,
                    };
                    edges.push(edge);
                }
            }
        }

        return {nodes, edges};
    }

    construct_map(edges) {
        const res = {};
        for (const ei in edges) {
            const edge = edges[ei];
            const {
                source,
                target,
                value,
                count,
            } = edge;
            const key = `${source},${target}`;
            res[key] = {count, rank: parseInt(ei)};      
        }

        return res;
    }

    construct_table() {
        const {data, prev_data} = this.props;
        const graph_data = this.construct_co_occurence_graph(data);
        let {edges} = this.construct_force_directed_data(graph_data);
        const {id_entity,} = graph_data;
        const sort_func = function(a, b){return b.count - a.count;};
        edges.sort(sort_func);
        const prev_graph_data = this.construct_co_occurence_graph(prev_data);
        const prev_graph = this.construct_force_directed_data(prev_graph_data);
        let prev_edges = prev_graph.edges;
        prev_edges.sort(sort_func);
        const prev_map = this.construct_map(prev_graph.edges);

        const table_rows = edges.map(({source, target, count}, index) => {
            const source_name = id_entity[source];
            const target_name = id_entity[target];
            let prev_count = null;
            let prev_rank = null;
            if (is_valid(prev_graph_data)) {
                const s_id = prev_graph_data.entity_id[source_name];
                const t_id = prev_graph_data.entity_id[target_name];
                if (!(is_valid(s_id) && is_valid(t_id))) {
                    prev_count = 0;
                } else {
                    prev_count = s_id in prev_graph_data.graph ?
                        prev_graph_data.graph[s_id].targets[t_id] :
                        null;
                    prev_rank = prev_map[`${s_id},${t_id}`];
                    prev_rank = is_valid(prev_rank) ? prev_rank.rank : null;
                }
            }

            return {
                id: `${source_name}, ${target_name}`,
                count,
                prev_count,
                rank: parseInt(index) + 1,
                prev_rank: is_valid(prev_rank) ? prev_rank + 1 : null,
            };
        });

        const table = <SortableTable 
            row_header={[
                {
                    id: 'id',
                    numeric: false,
                    disablePadding: false,
                    label: 'Entities'
                },
                {
                    id: 'rank',
                    numeric: true,
                    disablePadding: false,
                    label: 'Rank',
                },
                {
                    id: 'prev_rank',
                    numeric: true,
                    disablePadding: false,
                    label: 'Previous Rank',
                },
                {
                    id: 'count',
                    numeric: true,
                    disablePadding: false,
                    label: 'Count',
                },
                {
                    id: 'prev_count',
                    numeric: true,
                    disablePadding: false,
                    label: 'Previous Count',
                },
            ]}
            data={table_rows}
            defaultOrder={'desc'}
            defaultOrderBy={'count'}
        />;

        return table;
    }

    construct_ranked_list() {
        const {classes, data, prev_data} = this.props;
        const graph_data = this.construct_co_occurence_graph(data);
        let {edges} = this.construct_force_directed_data(graph_data);
        const {id_entity,} = graph_data;
        const sort_func = function(a, b){return b.count - a.count;};
        edges.sort(sort_func);
        
        const prev_graph_data = this.construct_co_occurence_graph(prev_data);
        return <List className={classes.list} dense={true}>
            {edges.map(({source, target, count}) => {
                const source_name = id_entity[source];
                const target_name = id_entity[target];
                let prev_count = null;
                if (is_valid(prev_graph_data)) {
                    const s_id = prev_graph_data.entity_id[source_name];
                    const t_id = prev_graph_data.entity_id[target_name];
                    if (!(is_valid(s_id) && is_valid(t_id))) {
                        prev_count = 0;
                    } else {
                        prev_count = s_id in prev_graph_data.graph ?
                            prev_graph_data.graph[s_id].targets[t_id] :
                            null;
                    }
                }
                return <span>
                    <ListItem>
                        <ListItemText
                            primary={`${source_name}, ${target_name}`}
                            secondary={`${count} | ${prev_count}`}
                        />
                        </ListItem>
                    <Divider />
                </span>;
            },
            )}
        </List>;
    }

    construct_edge_bundling_graph() {
        const graph_data = this.construct_co_occurence_graph();
        let entity_id, id_entity, graph = null;
        ({entity_id, id_entity, graph} = graph_data);
        const data = [];
        const deps = [];
        let parent = 0;
        for (const source in graph) {
            const targets = graph[source];
            const node_data = {
                name: id_entity[source],
                id: source,
            };

            if (parent !== -1) {
                node_data["parent"] = parent;
            }
            data.push(node_data);
            for (const ti in targets) {
                const target = targets[ti];
                const edge = {
                    source,
                    target,
                };
                deps.push(edge);
            }
            parent++;
        }

        return {data, deps};
    }

    render() {
        // const { dataset_id, classifier_class, data, show_predictions, show_labels} = this.props;
        // const {nodes, edges} = (show_predictions || show_labels) ? 
        //     construct_force_directed_data(data, null, show_predictions) :
        //     {nodes: null, edges: null};
        // const {data, deps} = this.construct_edge_bundling_graph();
        // return <div>
        //     {/* <GraphVis
        //         dataset_id={dataset_id}
        //         classifier_class={classifier_class}
        //         data={{nodes, edges}}
        //     /> */}
        //     {/* <EdgeBundleVis
        //         dataset_id={dataset_id}
        //         classifier_class={classifier_class}
        //         data={{data, deps}}
        //     /> */}
        //     <MatrixVis
        //         dataset_id={dataset_id}
        //         classifier_class={classifier_class}
        //         data={{nodes, edges}}
        //         ent_type={show_predictions ? 'Predicted' : 'Labeled'}
        //     />
        // </div>;
        // return this.construct_ranked_list();
        return this.construct_table();
    }
}

EntityCoOcurrence.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object,
    prev_data: PropTypes.object,
    show_predictions: PropTypes.bool,
    show_labels: PropTypes.bool,
};

export default withRoot(withStyles(styles)(EntityCoOcurrence));