import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
import withRoot from '../withRoot';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import VegaLite from 'react-vega-lite';
import EditableVisualization from '../visualizations/editable_visualization';
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
});

const POINT_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v2.0.json",
      "title": {
        "text": "Output Score Distribution",
        "anchor": "middle"
      },
      "description": "A histogram of the output score distribution",
      "width": 800,
      "height": 400,
      "mark": "point",
      "transform": [
            {"filter": "datum.contains_ent === true"}
        ],
      "encoding": {
        "x": {
          "field": "state",
          "type": "nominal"
        },
        "y": {
          "field": "ent_sent_score",
          "type": "quantitative",
          "scale": {"domain": [-1, 1]},
        },
        // "color" : {
        //     "field": "contains_ent",
        //     "type": "nominal",
        // }
      }
};

const MAP_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
    "width": 800,
    "height": 400,
    "layer": [
      {
        "data": {
          "url": "https://raw.githubusercontent.com/vega/vega/master/docs/data/us-10m.json",
          "format": {
            "type": "topojson",
            "feature": "states"
          }
        },
        "projection": {
          "type": "albersUsa"
        },
        "mark": {
          "type": "geoshape",
          "fill": "lightgray",
          "stroke": "white"
        }
      },
      {
        "projection": {
          "type": "albersUsa"
        },
        "mark": "circle",
        "transform": [
            {"filter": "datum.contains_ent === true"}
        ],
        "encoding": {
          "longitude": {
            "field": "long",
            "type": "quantitative"
          },
          "latitude": {
            "field": "lat",
            "type": "quantitative"
          },
          "size": {"value": 40},
          "color": {"field": "ent_sent_score", "type": "quantitative", "scale": {"range": "diverging"}}
        }
      }
    ]
  }

function is_overlapping(a, b) {
    // a => list[start, end]
    // b => list[start, end]
    // a ......
    // b         .....
    if (a[0] > b[0]) {
        const temp = a;
        a = b;
        b = temp;
    }

    // a is first
    if (a[1] >= b[0]) {
        return false;
    }
    return true;
}

class ResultVisualization extends React.Component {

    state = {
      datasource_value: 'graph',
      selector_open: false,
    };


    handleChange(event) {
      this.setState({datasource_value: event.target.value});
    }

    handleClose() {
        this.setState({
            selector_open: false,
        });
    }

    handleOpen() {
        this.setState({
            selector_open: true,
        });
    }


    process_data(data) {
        const convert_entire_data = [];
        for (const key in data.entire_data) {
            const curr = data.entire_data[key];
            const entry_id = parseInt(key);
            const contains_ent = false;
            const pred_data_entires = data.entry_to_sentences[entry_id];
            const ent_sent_score = 0;
            const ent_sent_score_total = 0;
            for (let i = 0; i < pred_data_entires.length; i++) {
                const index = pred_data_entires[i];
                const sentence_data = data.sentence_data[index];
                const entity_ranges = data.predictions[index][1].ranges;
                const sentence_spans = sentence_data[0];
                const sentence_sents = sentence_data[1];

                for (const eri in entity_ranges) {
                    const curr_er = entity_ranges[eri];

                    // check if overlaps
                    // if overlaps
                    // include that sentence in the ent_sent_score
                    for (const ssi in sentence_spans) {
                        const curr_ss = sentence_spans[ssi];
                        const s_sent = sentence_sents[ssi];
                        if (is_overlapping(curr_er, curr_ss)) {
                            ent_sent_score += s_sent.compound;
                            ent_sent_score_total += 1;
                        }
                    }
                }


                if (data.predictions[index][1].entities.length > 0) {
                    contains_ent = true;
                    break;
                }
            }
            const entry = {
                lat: parseFloat(curr['b.latitude']),
                long: parseFloat(curr['b.longitude']),
                stars: parseFloat(curr['r.stars']),
                sent: parseFloat(curr['sent_compound_score']),
                state: curr['b.state'],
                contains_ent: contains_ent,
                ent_sent_score: ent_sent_score_total > 0 ? ent_sent_score / ent_sent_score_total : 0,
            }
            convert_entire_data.push(entry);
        }
        return convert_entire_data;
    }

    getDatasource() {
      const {data, show_predictions} = this.props;
      const {datasource_value} = this.state;
      if (datasource_value == 'raw') {
        return {values: data};
      } else if (datasource_value == 'graph') {
        const {nodes, edges} = construct_force_directed_data(data.predictions, null, show_predictions);
        return {nodes, edges};
      }
    }

    render() {
        const {dataset_id, classifier_class, show_predictions, show_labels} = this.props;
        const {datasource_value, selector_open} = this.state;
        const datasource = this.getDatasource();
        if (!(show_predictions || show_labels)) {
          return null;
        }
        return (
            <div>
                <EditableVisualization
                    dataset_id={dataset_id}
                    classifier_class={classifier_class}
                    data={datasource}
                    dialogControls={ 
                    <FormControl style={{width: 150, margin: 10}}>
                        <InputLabel htmlFor="demo-controlled-open-select">Data Source</InputLabel>
                        <Select
                          open={selector_open}
                          onClose={this.handleClose.bind(this)}
                          onOpen={this.handleOpen.bind(this)}
                          value={datasource_value}
                          onChange={this.handleChange.bind(this)}
                          inputProps={{
                              name: 'datasodatasource_valueurce',
                              id: 'demo-controlled-open-select',
                          }}
                        >
                          <MenuItem value={'graph'}>Graph Data</MenuItem>
                          <MenuItem value={'raw'}>Raw Data</MenuItem>
                        </Select>
                    </FormControl>
                    }
                />
            </div>
        );
    }
}

ResultVisualization.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object,
    show_predictions: PropTypes.bool,
    show_labels: PropTypes.bool,
};

export default withRoot(withStyles(styles)(ResultVisualization));