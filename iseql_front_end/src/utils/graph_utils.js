import {is_valid} from '../utils';

export function construct_co_occurence_graph(data, use_predictions) {
    if (!is_valid(data)) {
        return null;
    }
    const id_entity = [];
    const entity_id = {};
    const graph = {}
    for (const s_id in data) {
        const entry_data = data[s_id];
        const sentence_data = entry_data[1];
        const entities = use_predictions ? sentence_data.entities : sentence_data.real_entities;
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

export function construct_force_directed_data(data, graph_data=null, use_predictions=false) {
    graph_data = graph_data === null ? construct_co_occurence_graph(data, use_predictions) : graph_data;
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