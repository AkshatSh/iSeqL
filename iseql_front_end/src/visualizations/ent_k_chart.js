export function construct_spec(ent_data) {
    return {
        "$schema": "https://vega.github.io/schema/vega/v4.json",
        "width": 500,
        "height": 410,
        "padding": 5,
        "autosize": "fit",

        "title":  {"text": "Top Entities"},
      
        "signals": [
          {
            "name": "k", "value": 20,
            "bind": {"input": "range", "min": 10, "max": 30, "step": 1}
          }
        ],
      
        "data": [
                  {
                    "name": "table",
                    "values": ent_data,
                    "transform": [
                      {
                        "type": "aggregate",
                        "groupby": ["ent"]
                      },
                      {
                        "type": "window",
                        "sort": {"field": ["count", "ent"], "order": ["descending", "ascending"]},
                        "ops": ["row_number"], "as": ["rank"]
                      },
                      {
                        "type": "formula",
                        "as": "Category",
                        "expr": "datum.rank < k ? datum.ent : null"
                      },
                      {"type": "filter", "expr": "datum.Category !== null"},
                      {
                        "type": "aggregate",
                        "groupby": ["Category"],
                        "ops": ["average"],
                        "fields": ["count"],
                        "as": ["total_count"]
                      }
                    ]
                  }
                ],
      
        "marks": [
          {
            "type": "rect",
            "from": {"data": "table"},
            "encode": {
              "update": {
                "x": {"scale": "x", "value": 0},
                "x2": {"scale": "x", "field": "total_count"},
                "y": {"scale": "y", "field": "Category"},
                "height": {"scale": "y", "band": 1}
              }
            }
          }
        ],
      
        "scales": [
          {
            "name": "x",
            "type": "linear",
            "domain": {"data": "table", "field": "total_count"},
            "range": "width",
            "nice": true
          },
          {
            "name": "y",
            "type": "band",
            "domain": {
              "data": "table", "field": "Category",
              "sort": {"op": "max", "field": "total_count", "order": "descending"}
            },
            "range": "height",
            "padding": 0.1
          }
        ],
      
        "axes": [
          {
            "scale": "x",
            "orient": "bottom",
            "format": "d",
            "tickCount": 5
          },
          {
            "scale": "y",
            "orient": "left"
          }
        ]
      };
}