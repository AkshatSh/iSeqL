/*jshint esversion: 6 */

import React, { Component } from "react";
import Button from '@material-ui/core/Button';
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import VegaLite from 'react-vega-lite';
import VegaLiteDialog from './VegaLiteDialog';
import Vega from 'react-vega';
import {construct_spec} from './matrix_graph_spec';

const InitialJsCode = `function process_data(data) {
  // manipulate data here
  return data;
}
`;

const InitialViz = {
    "$schema": "https://vega.github.io/schema/vega-lite/v2.0.json",
      "title": {
        "text": "Output Score Distribution",
        "anchor": "middle"
      },
      "description": "A histogram of the output score distribution",
      "width": 800,
      "height": 400,
      "mark": "bar",
      "encoding": {
        "x": {
          "bin": {"maxbins": 30},
          "field": "output",
          "type": "quantitative"
        },
        "y": {
          "aggregate": "count",
          "type": "quantitative"
        }
      }
};
 
class EditableVisualization extends Component {
  constructor(props) {
    super(props);

    this.state = {
        js_code: InitialJsCode,
        spec: construct_spec('Entity Co-occurence Matrix'),
        shown: false,
        vega_type: 'Vega',
        selector_open: false,
    };
  }

  // componentDidMount() {
  //   const convert_entire_data = [];
  //   for (const key in this.props.data.entire_data) {
  //     const curr = this.props.data.entire_data[key];
  //     const entry = {
  //       lat: parseFloat(curr['b.latitude']),
  //       long: parseFloat(curr['b.longitude']),
  //       stars: parseFloat(curr['r.stars']),
  //       sent: parseFloat(curr['sent_compound_score']),
  //     }
  //     convert_entire_data.push(entry);
  //   }

  //   this.setState({values: convert_entire_data});
  // }

  // componentWillReceiveProps(oldProps, newProps) {

  //   const convert_entire_data = [];
  //   for (const key in newProps.data.entire_data) {
  //     const curr = newProps.data.entire_data[key];
  //     const entry = {
  //       lat: parseFloat(curr['b.latitude']),
  //       long: parseFloat(curr['b.longitude']),
  //       stars: parseFloat(curr['r.stars']),
  //       sent: parseFloat(curr['sent_compound_score']),
  //     }
  //     convert_entire_data.push(entry);
  //   }

  //   this.setState({values: convert_entire_data});
  // }


  handleDialogClose(spec) {
      this.setState({spec: spec, shown: false});
  }

  handleClickOpen() {
      this.setState({shown: true});
  }

  handleChange(event) {
      this.setState({
          vega_type: event.target.value,
      });
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

  setVegaType(vega_type) {
    this.setState({vega_type});
  }
    
  render() {
    if (this.state.spec === null) {
        // error message 
        return <div>Invalid Vega Lite Spec</div>;
    } else {
        const {spec, shown, js_code, vega_type, selector_open} = this.state;
        const {data, dialogControls} = this.props;
        return <div>
            <div>
              <Button
                  variant="contained"
                  color="#9e9e9e"
                  onClick={this.handleClickOpen.bind(this)}>
                  Edit {vega_type} Spec
              </Button>
            </div>
                <VegaLiteDialog
                    defaultSpec={spec}
                    defaultJSCode={js_code}
                    open={shown}
                    onClose={this.handleDialogClose.bind(this)}
                    setVegaType={this.setVegaType.bind(this)}
                >
                  <form autoComplete="off">
                    {dialogControls}
                </form>
                </VegaLiteDialog>
            {vega_type == 'Vega' ? <Vega spec={spec} data={data}/> : <VegaLite spec={spec} data={data} />}
        </div>
    }
  }
}
 
export default EditableVisualization;