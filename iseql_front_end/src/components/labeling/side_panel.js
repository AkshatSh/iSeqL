import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import { withStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import DialogTitle from '@material-ui/core/DialogTitle';
import Dialog from '@material-ui/core/Dialog';
import Button from '@material-ui/core/Button';
import Select from '@material-ui/core/Select';
import LabelProgress from './label_progress';
import TrainSummary from './train_summary';
import SurprsingLabels from './surprising_labels';
import withRoot from '../../withRoot';
import {is_valid} from '../../utils';

const styles = theme => ({
root: {
    textAlign: 'center',
    paddingTop: theme.spacing.unit * 20,
},
paper_internal: {
    // padding: theme.spacing.unit * 2,
    textAlign: "center",
    // minWidth: theme.spacing.unit * 40,
    width: theme.spacing.unit * 40,
    height: theme.spacing.unit * 80,
    marginLeft: theme.spacing.unit * 2,
    top: theme.spacing.unit * 10,
    position: "fixed",
    right: theme.spacing.unit * 2,
    border: "1px solid black",
},
formControl: {
    margin: theme.spacing.unit,
    minWidth: 120,
},
dialog_internal: {
    padding: theme.spacing.unit * 2,
    textAlign: "center",
    // minWidth: theme.spacing.unit * 400,
}
});

class SidePanel extends React.Component {
    state = {
        selected_mode: 1,
        selector_open: false,
        dialog_open: false,
    };

    handleChange(event) {
        this.setState({
            selected_mode: event.target.value,
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

    create_selector() {
        const {classes} = this.props;
        const {selected_mode, selector_open} = this.state;
        return (
            <form autoComplete="off">
                <FormControl className={classes.formControl}>
                    <InputLabel htmlFor="demo-controlled-open-select">Mode</InputLabel>
                    <Select
                    open={selector_open}
                    onClose={this.handleClose.bind(this)}
                    onOpen={this.handleOpen.bind(this)}
                    value={selected_mode}
                    onChange={this.handleChange.bind(this)}
                    inputProps={{
                        name: 'selected_mode',
                        id: 'demo-controlled-open-select',
                    }}
                    >
                    <MenuItem value={1}>Label Progress</MenuItem>
                    <MenuItem value={2}>Train Summary</MenuItem>
                    <MenuItem value={3}>Unlabeled Entities</MenuItem>
                    </Select>
                </FormControl>
            </form>
        );
    }

    handleDialogOpen() {
        this.setState({dialog_open: true});
    }
    handleDialogClose() {
        this.setState({dialog_open: false});
    }

    create_dialog() {
        const {
            classes,
            flipped_data,
            train_summary,
            labeled_set_sizes,
            predictions,
        } = this.props;

        const {selected_mode, dialog_open} = this.state;
        return (
            <Dialog onClose={this.handleDialogClose.bind(this)} aria-labelledby="vega-lite-spec" open={dialog_open}>
              <DialogTitle id="dialog-labeling-progress" style={{textAlign: "center"}}>Labeling Progress</DialogTitle>
              <Paper style={{minWidth: '600px'}}>
                    <div className={classes.dialog_internal}>
                        {selected_mode === 1 ? <LabelProgress
                            flipped_data={flipped_data}
                        /> : null}
                        {selected_mode === 2 ?<TrainSummary
                            train_summary={train_summary}
                            labeled_set_sizes={labeled_set_sizes}
                        /> : null}
                        {selected_mode === 3 ? <SurprsingLabels
                            predictions={predictions}
                            height={"400px"}
                        /> : null}
                    </div>
                </Paper>
            </Dialog>
        );
    }

    empty_data() {
        return <Typography style={{top: "50%", textAlign: "center", position: "relative"}} variant="h6">
            No Data
        </Typography>;
    }

    render() {
        const {
            classes,
            flipped_data,
            train_summary,
            labeled_set_sizes,
            predictions,
            header,
        } = this.props;

        return (
            <div>
                <Paper className={classes.paper_internal}>
                    {header}
                    {labeled_set_sizes.length === 0 ? 
                        this.empty_data()
                    :
                    <div>
                        <div style={{borderTop: "1px solid black", borderBottom: "1px solid black"}}>
                        <LabelProgress
                            flipped_data={flipped_data}
                        />
                        </div>
                        <div style={{
                            borderBottom: "1px solid black",
                            // borderLeft: "1px solid black",
                            // borderRight: "1px solid black",
                        }}>
                        <TrainSummary
                            style={{border: "1px solid black"}}
                            train_summary={train_summary}
                            labeled_set_sizes={labeled_set_sizes}
                        />
                        </div>
                        <div style={{
                            // borderBottom: "1px solid black",
                            // borderLeft: "1px solid black",
                            // borderRight: "1px solid black",
                        }}>
                        <SurprsingLabels
                            style={{border: "1px solid black"}}
                            predictions={predictions}
                            height={"180px"}
                            updateTopEntities={this.props.updateTopEntities}
                        />
                        </div>
                    </div>
                }
                </Paper>
            </div>
        );
    }
}

SidePanel.propTypes = {
    header: PropTypes.node,
    data: PropTypes.object,
    flipped_data: PropTypes.array,
    predictions: PropTypes.object,
    train_summary: PropTypes.array,
    labeled_set_sizes: PropTypes.array,
    updateTopEntities: PropTypes.func,
};

export default withRoot(withStyles(styles)(SidePanel));