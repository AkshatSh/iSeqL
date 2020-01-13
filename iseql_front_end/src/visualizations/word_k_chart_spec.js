export function construct_spec(word_data) {
    return {
        "$schema": "https://vega.github.io/schema/vega/v4.json",
        "width": 500,
        "height": 410,
        "padding": 5,
        "autosize": "fit",

        "title":  {"text": "Top Words in Entities"},
      
        "signals": [
          {
            "name": "k", "value": 20,
            "bind": {"input": "range", "min": 10, "max": 30, "step": 1}
          },
        ],
      
        "data": [
                  {
                    "name": "table",
                    "values": [word_data],
                    "transform": [
                      {
                        "type": "countpattern",
                        "field": "data",
                        "case": "upper",
                        "pattern": "[\\w']{3,}",
                        "stopwords": "(i|me|my|myself|we|us|our|ours|ourselves|you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|whose|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|will|would|should|can|could|ought|i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|i'd|you'd|he'd|she'd|we'd|they'd|i'll|you'll|he'll|she'll|we'll|they'll|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|shan't|shouldn't|can't|cannot|couldn't|mustn't|let's|that's|who's|what's|here's|there's|when's|where's|why's|how's|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|upon|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|say|says|said|shall)"
                      },
                      {
                        "type": "window",
                        "sort": {"field": "count", "order": "descending"},
                        "ops": ["row_number"], "as": ["rank"]
                      },
                      {
                        "type": "formula",
                        "as": "Category",
                        "expr": "datum.rank < k ? datum.text : null"
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