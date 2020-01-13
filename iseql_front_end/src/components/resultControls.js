import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import FormLabel from '@material-ui/core/FormLabel';
import FormControl from '@material-ui/core/FormControl';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Button from '@material-ui/core/Button';
import Switch from '@material-ui/core/Switch';
import Checkbox from '@material-ui/core/Checkbox';
import MobileStepper from '@material-ui/core/MobileStepper';
import KeyboardArrowLeft from '@material-ui/icons/KeyboardArrowLeft';
import KeyboardArrowRight from '@material-ui/icons/KeyboardArrowRight';
import withRoot from '../withRoot';
import is_valid from '../utils';

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
MobileStepper: {
    textAlign: 'center',
}
});

class ResultControls extends React.Component {

    state = {
        train: true,
        test: true,
        unlabeled: true,
        show_predictions: false,
        combined_view: false,
        show_labels: true,
        show_predictions: false,
        activeStep: 0,
    }

    handleChange = name => event => {
        const {onControlsUpdate} = this.props;
        this.setState({ ...this.state, [name]: event.target.checked });
        onControlsUpdate(name, event.target.checked);
    };

    render() {
        const {train, test, unlabeled, show_labels, show_predictions} = this.state;
        const {classes, evaluateModelFunc, fetchPredictionsFunc, theme, numberOfTrainIters} = this.props;
        return (
            <div>
                <FormControl component="fieldset" className={classes.formControl}>
                    <FormGroup style ={{display: "inline"}}>
                        <FormControlLabel
                            control={<Checkbox checked={train} onChange={this.handleChange('train').bind(this)} value="train" />}
                            label="Train"
                        />
                        <FormControlLabel
                            control={<Checkbox checked={test} onChange={this.handleChange('test').bind(this)} value="test" />}
                            label="Test"
                        />
                        <FormControlLabel
                            control={
                            <Checkbox checked={unlabeled} onChange={this.handleChange('unlabeled').bind(this)} value="unlabeled" />
                            }
                            label="Unlabeled"
                        />
                        <FormControlLabel
                            control={
                            <Checkbox checked={show_labels} onChange={this.handleChange('show_labels').bind(this)} value="show_labels" />
                            }
                            label="Show Labels"
                        />
                        <FormControlLabel
                            control={
                            <Checkbox checked={show_predictions} onChange={this.handleChange('show_predictions').bind(this)} value="show_predictions" />
                            }
                            label="Show Predictions"
                        />
                        {/* {is_valid(fetchPredictionsFunc) ?<Button
                            variant="contained"
                            color="primary"
                            onClick={fetchPredictionsFunc}
                            className={classes.fab}>
                            Fetch Predictions
                        </Button>: null} */}
                    </FormGroup>
                </FormControl>
            </div>
        );
    }
}

ResultControls.propTypes = {
    classes: PropTypes.object.isRequired,
    theme: PropTypes.object.isRequired,
    onControlsUpdate: PropTypes.func,
    fetchPredictionsFunc: PropTypes.func,
};

export default withRoot(withStyles(styles, { withTheme: true })(ResultControls));