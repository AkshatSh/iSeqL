export function construct_spec(title) {
    return {
        "$schema": "https://vega.github.io/schema/vega/v4.json",
        "width": 770,
        "height": 770,
        "padding": 2,

        "title": title,

        "legends": [
          {
            "fill": "color",
            "encode": {
              "title": {
                "update": {
                  "fontSize": {"value": 14}
                }
              },
              "labels": {
                "update": {
                  "fontSize": {"value": 12},
                  "fill": {"value": "black"}
                }
              }
            }
          }
        ],
      
        "signals": [
        {
            "name": "k", "value": 50,//4,
            // "bind": {"input": "range", "min": 0, "max": 10, "step": 1}
            "bind": {"input": "range", "min": 0, "max": 100, "step": 1}
            },
          { "name": "cellSize", "value": 10 },
          { "name": "count", "update": "length(data('nodes'))" },
          { "name": "width", "update": "span(range('position'))" },
          { "name": "height", "update": "width" },
          {
            "name": "src", "value": {},
            "on": [
              {"events": "text:mousedown", "update": "datum"},
              {"events": "window:mouseup", "update": "{}"}
            ]
          },
          {
            "name": "dest", "value": -1,
            "on": [
              {
                "events": "[@columns:mousedown, window:mouseup] > window:mousemove",
                "update": "src.name && datum !== src ? (0.5 + count * clamp(x(), 0, width) / width) : dest"
              },
              {
                "events": "[@rows:mousedown, window:mouseup] > window:mousemove",
                "update": "src.name && datum !== src ? (0.5 + count * clamp(y(), 0, height) / height) : dest"
              },
              {"events": "window:mouseup", "update": "-1"}
            ]
          }
        ],
      
        "data": [
          {
            "name": "nodes",
            // "values": nodes,
            "transform": [
              {
                "type": "formula", "as": "order",
                "expr": "datum.group"
              },
              {
                "type": "formula", "as": "score",
                "expr": "dest >= 0 && datum === src ? dest : datum.order"
              },
              {
                "type": "window", "sort": {"field": "score"},
                "ops": ["row_number"], "as": ["order"]
              },
              {
                "type": "window", "sort": {"field": "count", "order": "descending"},
                "ops": ["row_number"], "as": ["sort_order"]
              },
              // {"type": "filter", "expr": "datum.count >= k"},
              {"type": "filter", "expr": "datum.sort_order <= k"},
            ]
          },
          {
            "name": "edges",
            // "values": edges,
            "transform": [
              {
                "type": "lookup", "from": "nodes", "key": "index",
                "fields": ["source", "target"], "as": ["sourceNode", "targetNode"]
              },
              {
                  "type": "filter",
                  "expr": "datum.sourceNode !== null && datum.targetNode !== null"
              },
              {
                "type": "formula", "as": "group",
                "expr": "datum.sourceNode.group === datum.targetNode.group ? datum.sourceNode.group : count"
              },
            ]
          },
          {
            "name": "cross",
            "source": "nodes",
            "transform": [
              { "type": "cross" }
            ]
          }
        ],
      
        "scales": [
          {
            "name": "position",
            "type": "band",
            "domain": {"data": "nodes", "field": "order", "sort": true},
            "range": {"step": {"signal": "cellSize"}}
          },
          {
            "name": "color",
            "type": "sequential",
            "range": "ordinal",
            "domain": {
              "fields": [
                {"data": "edges", "field": "count"},
                {"signal": "count"}
              ],
              "sort": true
            }
          }
        ],
      
        "marks": [
          {
            "type": "rect",
            "from": {"data": "cross"},
            "encode": {
              "update": {
                "x": {"scale": "position", "field": "a.order"},
                "y": {"scale": "position", "field": "b.order"},
                "width": {"scale": "position", "band": 1, "offset": -1},
                "height": {"scale": "position", "band": 1, "offset": -1},
                "fill": [
                  {"test": "datum.a === src || datum.b === src", "value": "#ddd"},
                  {"value": "#f5f5f5"}
                ]
              }
            }
          },
          {
            "type": "rect",
            "from": {"data": "edges"},
            "encode": {
              "update": {
                "x": {"scale": "position", "field": "sourceNode.order"},
                "y": {"scale": "position", "field": "targetNode.order"},
                "width": {"scale": "position", "band": 1, "offset": -1},
                "height": {"scale": "position", "band": 1, "offset": -1},
                "fill": {"scale": "color", "field": "count"}
              }
            }
          },
          {
            "type": "rect",
            "from": {"data": "edges"},
            "encode": {
              "update": {
                "x": {"scale": "position", "field": "targetNode.order"},
                "y": {"scale": "position", "field": "sourceNode.order"},
                "width": {"scale": "position", "band": 1, "offset": -1},
                "height": {"scale": "position", "band": 1, "offset": -1},
                "fill": {"scale": "color", "field": "count"}
              }
            }
          },
          {
            "type": "text",
            "name": "columns",
            "from": {"data": "nodes"},
            "encode": {
              "update": {
                "x": {"scale": "position", "field": "order", "band": 0.5},
                "y": {"offset": -2},
                "text": {"field": "name"},
                "fontSize": {"value": 10},
                "angle": {"value": -90},
                "align": {"value": "left"},
                "baseline": {"value": "middle"},
                "fill": [
                  {"test": "datum === src", "value": "steelblue"},
                  {"value": "black"}
                ]
              }
            }
          },
          {
            "type": "text",
            "name": "rows",
            "from": {"data": "nodes"},
            "encode": {
              "update": {
                "x": {"offset": -2},
                "y": {"scale": "position", "field": "order", "band": 0.5},
                "text": {"field": "name"},
                "fontSize": {"value": 10},
                "align": {"value": "right"},
                "baseline": {"value": "middle"},
                "fill": [
                  {"test": "datum === src", "value": "steelblue"},
                  {"value": "black"}
                ]
              }
            }
          }
        ]
      };
}
