import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import { withStyles } from '@material-ui/core/styles';
import CircularProgress from '@material-ui/core/CircularProgress';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import LinearProgress from '@material-ui/core/LinearProgress';
import withRoot from '../withRoot';
import configuration from '../configuration';
import TrainingProgress from './training_progress';
import {is_valid} from '../utils';

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
progress: {
    margin: theme.spacing.unit * 2,
    display: "block",
},
paper: {
    ...theme.mixins.gutters(),
    paddingTop: theme.spacing.unit * 2,
    paddingBottom: theme.spacing.unit * 2,
    marginTop: theme.spacing.unit * 2,
},
grow: {
    flexGrow: 1,
},
});

class ModelTrainer extends React.Component {
    state = {
        running: false,
        thread_id: -1,
        training_progress: {},
        progress_message: "",
        expanded: "",
    };

    handlePanelChange = panel => (event, isExpanded) => {
        this.setState({expanded:  isExpanded ? panel : false});
    };

    train_on_click() {
        
    }

    render() {
        const {classes, classifier_class, start_train, training_progress} = this.props;
        const {expanded, running,} = this.state;
        const training_progress_data = is_valid(training_progress) ? training_progress.train_progress : null;
        const total_epochs = is_valid(training_progress) ? training_progress.num_epochs : null;
        let progress_message = null;

        if (is_valid(training_progress_data)) {
            progress_message = `Training Epoch: ${training_progress_data.length}/${total_epochs}`;
            if (training_progress_data.length >= total_epochs) {
                progress_message = "Evaluating Model On All Data ...";
            }
        } else {
            return null;
        }

        return (
            <ExpansionPanel
                expanded={expanded === 'panel1'}
                onChange={this.handlePanelChange('panel1').bind(this)}
            >
                <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h4" gutterBottom>Model Training</Typography>
                </ExpansionPanelSummary>
                <ExpansionPanelDetails>
                {
                    !is_valid(training_progress_data) ? null :
                    <div>
                    <Typography>
                        {progress_message}
                    </Typography>
                    <LinearProgress
                        color="secondary"
                        variant="determinate"
                        value={training_progress_data.length * 100.0 / total_epochs}
                        className={classes.grow}
                    />
                    <TrainingProgress
                        training_progress_data={training_progress_data}
                        classifier_class={classifier_class}
                    />
                    </div>
                }
                </ExpansionPanelDetails>
            </ExpansionPanel>
        );
    }
}

ModelTrainer.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    start_train: PropTypes.bool,
    training_progress: PropTypes.object,
};

export default withRoot(withStyles(styles)(ModelTrainer));