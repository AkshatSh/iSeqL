/*jshint esversion: 6 */

import React, { Component } from "react";
import DialogTitle from '@material-ui/core/DialogTitle';
import Dialog from '@material-ui/core/Dialog';
import JSONInput from 'react-json-editor-ajrm';
import locale from 'react-json-editor-ajrm/locale/en';
import Button from '@material-ui/core/Button';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';

// import "brace/mode/javascript";
// import "brace/mode/json";
// import "brace/theme/eclipse";

    // <JSONInput
    //     id          = 'a_unique_id'
    //     placeholder = { sampleObject }
    //     colors      = { darktheme }
    //     locale      = { locale }
    //     height      = '550px'
    // />

const styles = theme => ({
    container: {
        display: "flex",
    },

    js_code_editor: {
        flexGrow: 1,
    },

    vega_editor: {
        flexGrow: 1,
    }
});

// A component for a popup dialog that a user uses to enter a vega lite spec
// Props:
//   onClose: Takes a function for onClose that has a parameter of the vegalite spec
//   defaultSpec: The default spec for the input
//   open: whether to show the dialog
class VegaLiteDialog extends Component {
    constructor(props) {
        super(props);

        this.state = {
            currentSpec: this.props.defaultSpec,
            javascript_code: this.props.defaultJSCode,
            selector_open: false,
            vega_type: 'Vega',
        };
    }


    handleDialogClose() {
        this.props.onClose(this.state.currentSpec);
        this.props.setVegaType(this.state.vega_type);
    }

    updateSpec(obj) {
        this.setState({currentSpec: obj});
    }

    updateJavascriptTransform(code) {
        this.setState({javascript_code: code});
    }

    handleChange(event) {
        this.setState({
            vega_type: event.target.value,
            selector_open: false,
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

    render() {
        const {classes} = this.props;
        const {selector_open, vega_type} = this.state;
        return (
            <Dialog onClose={this.handleDialogClose.bind(this)} aria-labelledby="vega-lite-spec" open={this.props.open} maxWidth="xl">
                {/* <DialogTitle id="vega-lite-spec-title">Enter a Vega Lite Spec</DialogTitle> */}
                <div className={classes.container}>
                    <div className={classes.vega_editor}>
                        <DialogTitle id="vega-lite-spec-title">Enter a {vega_type} Spec</DialogTitle>
                        <div>
                            {this.props.children}
                            <FormControl style={{width: 150, margin: 10}}>
                                <InputLabel htmlFor="demo-controlled-open-select-vega">Vega Type</InputLabel>
                                <Select
                                    open={selector_open}
                                    onClose={this.handleClose.bind(this)}
                                    onOpen={this.handleOpen.bind(this)}
                                    value={vega_type}
                                    onChange={this.handleChange.bind(this)}
                                    inputProps={{
                                        name: 'vega_type',
                                        id: 'demo-controlled-open-select-vega',
                                    }}
                                >
                                    <MenuItem value={'Vega'}>Vega</MenuItem>
                                    <MenuItem value={'Vega-Lite'}>Vega-Lite</MenuItem>
                                </Select>
                            </FormControl>
                        </div>
                        <JSONInput
                            id = 'vega_lite_spec'
                            placeholder = {this.props.defaultSpec}
                            theme = "dark_vscode_tribute"
                            locale = {locale}
                            height = '500px'
                            width = '500px'
                            onChange = {
                                (event) => {
                                    this.updateSpec(event.jsObject);
                                }
                            }
                        />
                    </div>
                </div>
                <Button onClick={this.handleDialogClose.bind(this)}>Update</Button>
            </Dialog>
          );
    }
}

export default withRoot(withStyles(styles, {withTheme: true})(VegaLiteDialog));