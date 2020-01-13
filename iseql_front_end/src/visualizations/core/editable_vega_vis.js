/*jshint esversion: 6 */

import React, { Component } from "react";
import Button from '@material-ui/core/Button';
import VegaLite from 'react-vega-lite';
import VegaLiteDialog from '../VegaLiteDialog';

class EditableVegaSpec extends Component {
  constructor(props) {
    super(props);

    this.state = {
        spec: null,
        shown: false,
    };

    this.handleClickOpen = () => {
        this.setState({shown: true});
    };

    this.handleClose = spec => {
        this.setState({spec: spec, shown: false,});
    };
  }

  componentDidMount() {
      this.setState({spec: this.props.spec});
  }
    
  render() {
    if (this.state.spec === null) {
        // error message 
        return <div>Invalid Vega Lite Spec</div>;
    } else {
        return <div>
            <Button
                variant="contained"
                color="#9e9e9e"
                onClick={this.handleClickOpen}>
                Edit Vega Lite Spec
            </Button>
                <VegaLiteDialog
                    defaultSpec={this.state.spec}
                    open={this.state.shown}
                    onClose={this.handleClose}
                />
            <VegaLite spec={this.state.spec} data={{values: this.props.data}} style={{}}/>
        </div>
    }
  }
}
 
export default EditableVegaSpec;