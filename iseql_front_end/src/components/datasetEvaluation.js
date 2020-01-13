import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import MobileStepper from '@material-ui/core/MobileStepper';
import KeyboardArrowLeft from '@material-ui/icons/KeyboardArrowLeft';
import KeyboardArrowRight from '@material-ui/icons/KeyboardArrowRight';
import Paper from '@material-ui/core/Paper';
import Stepper from '@material-ui/core/Stepper';
import Step from '@material-ui/core/Step';
import StepButton from '@material-ui/core/StepButton';
import withRoot from '../withRoot';
import ResultExplorer from './resultExplorer';
import ResultTable from './resultTable';
import ResultStatistics from './resultStatistics';
import ResultVisualization from './resultVisualization';
import ResultControls from './resultControls';
import EntityCoOcurrence from './entityCooccurence';
import {ACTIVE_LEARNING_SERVER} from '../configuration';
import {get_user_url, is_valid} from '../utils';

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
heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
},
paddingEverything : {
    padding: theme.spacing.unit * 4,
},
});

class DatasetEvaluation extends React.Component {
    state = {
        predictions: {},
        training_set_sizes: [],
        train: true,
        test: true,
        unlabeled: true,
        combined_view: false,
        show_predictions: false,
        show_labels: true,
        activeStep: 0,
        trainsetStep: 0,
        entry_to_sentences: [],
        entire_data: [],
        sentence_data: [],
    };

    // evaluateClick() {
    //     const url = get_user_url(
    //         `${ACTIVE_LEARNING_SERVER}/api/evaluate`,
    //     );
    //     fetch(
    //         url,
    //     ).then(results => {
    //         return results.json();
    //     }).then(results => { 
    //         this.setState({
    //             predictions: results,
    //         });
    //     });
    // }

    onControlUpdate(name, value) {
        this.setState({ ...this.state, [name]: value });
    }

    fetchPredictions() {
        const url = get_user_url(
            `${ACTIVE_LEARNING_SERVER}/api/predictions`,
        );
        fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(results => {
            if (results.training_set_sizes.length !== this.state.training_set_sizes.length) {
                this.setState({
                    predictions: results.predicted_data,
                    training_set_sizes: results.training_set_sizes,
                    trainsetStep: results.training_set_sizes.length -1 >= 0 ? results.training_set_sizes.length -1 : 0,
                    entry_to_sentences: results.entry_to_sentences,
                    entire_data: results.entire_data,
                    sentence_data: results.sentence_data,
                });
            }
        });
    }

    componentWillReceiveProps(nextProps) {
        const results = nextProps.prediction_result;
        if (is_valid(results) && results.predictions != this.state.predictions) {
            this.setState({
                predictions: results.predicted_data,
                training_set_sizes: results.training_set_sizes,
                trainsetStep: results.training_set_sizes.length -1 >= 0 ? results.training_set_sizes.length -1 : 0,
                entry_to_sentences: results.entry_to_sentences,
                entire_data: results.entire_data,
                sentence_data: results.sentence_data,
            });
        }
    }

    filterResults() {
        const {predictions} = this.state;
        const {train, test, unlabeled, trainsetStep} = this.state;
        const result = {};
        const prev_results = {};
        for (const s_id in predictions) {
            const entry_data = predictions[s_id];
            const sentence_data = entry_data[1];

            const is_test = sentence_data.is_test;
            const is_train = sentence_data.is_train;
            const is_unlabeled = !is_test && !is_train;
            const should_render = (train && is_train) || (test && is_test) || (unlabeled && is_unlabeled);
            if (!should_render) {
                continue;
            }

            const new_entry_data = {};
            new_entry_data[0] = entry_data[0];
            new_entry_data[1] = Object.assign({}, entry_data[1]);
            new_entry_data[1].entities = new_entry_data[1].entities[trainsetStep];
            new_entry_data[1].ranges = new_entry_data[1].ranges[trainsetStep];

            if (trainsetStep > 0) {
                const prev_entry_data = {};
                prev_entry_data[0] = entry_data[0];
                prev_entry_data[1] = Object.assign({}, entry_data[1]);
                prev_entry_data[1].entities = prev_entry_data[1].entities[trainsetStep - 1];
                prev_entry_data[1].ranges = prev_entry_data[1].ranges[trainsetStep - 1];
                prev_results[s_id] = prev_entry_data;
            }
            result[s_id] = new_entry_data;
        }

        return {result, prev_results};
    }

    getSteps() {
        return ['Explore Datset', 'Prediction Statistics', 'Entity Co-occurence', 'Custom Visualization'];
    }
      
    getStepContent(step) {
        switch (step) {
            case 0:
                return 'View Dataset with Predictions';
            case 1:
                return 'View Result Statistics';
            case 2:
                return 'View Entity Co-occurence'
            case 3:
                return 'View Visualizations';
            default:
                return 'Unknown step';
        }
    }

    handleStep = index => (args) => {
        this.setState({
            activeStep: index,
        });
    }

    displaySection(index) {
        const {activeStep} = this.state;
        const res = index === activeStep ? "inherit" : "none";
        return res;
    }



    handleNext = () => {
        this.setState(state => ({
            trainsetStep: state.trainsetStep + 1,
        }));
    };
    
    handleBack = () => {
        this.setState(state => ({
            trainsetStep: state.trainsetStep - 1,
        }));
    };
      

    render() {
        const {theme, classes, dataset_id, classifier_class} = this.props;
        const {
            show_predictions,
            trainsetStep,
            training_set_sizes,
            activeStep,
            combined_view,
            entry_to_sentences,
            entire_data,
            sentence_data,
            show_labels,
        } = this.state;
        // this.fetchPredictions();
        const {result, prev_results} = this.filterResults();
        const predictions = result;
        const numberOfTrainIters = training_set_sizes === null ? 0 : training_set_sizes.length;
        const steps = this.getSteps();
        return (
            <div>
                <Paper className={classes.paddingEverything} style={{top: theme.spacing.unit * 10,}}>
                <Typography variant="h5" gutterBottom>
                Dataset Evaluation
                </Typography>
                <ResultControls
                    onControlsUpdate={this.onControlUpdate.bind(this)}
                    // fetchPredictionsFunc={this.fetchPredictions.bind(this)}
                    numberOfTrainIters={training_set_sizes === null ? 0 : training_set_sizes.length}
                />
                <Stepper alternativeLabel nonLinear activeStep={activeStep}>
                    {steps.map((label, index) => {
                    const stepProps = {};
                    const buttonProps = {};
                    return (
                        <Step key={label} {...stepProps}>
                        <StepButton
                            onClick={this.handleStep(index).bind(this)}
                            completed={false}
                            {...buttonProps}
                        >
                            {label}
                        </StepButton>
                        </Step>
                    );
                    })}
                </Stepper>
                <div style={{display: this.displaySection(0)}}>
                    {/* <ResultExplorer
                        dataset_id={dataset_id}
                        classifier_class={classifier_class}
                        data={predictions}
                        is_predicted={show_predictions}
                        combined_view={combined_view}
                        show_predicted={show_predictions || combined_view}
                        show_labeled={!show_predictions || combined_view}
                    /> */}
                    <ResultTable
                        dataset_id={dataset_id}
                        classifier_class={classifier_class}
                        data={predictions}
                        is_predicted={show_predictions}
                        combined_view={show_predictions && show_labels}
                        show_predicted={show_predictions}
                        show_labeled={show_labels}
                    />
                </div>
                <div style={{display: this.displaySection(1)}}>
                    <ResultStatistics
                        style={{display: this.displaySection(1)}}
                        dataset_id={dataset_id}
                        classifier_class={classifier_class}
                        data={predictions}
                        show_predictions={show_predictions}
                        show_labels={show_labels}
                    />
                </div>
                <div style={{display: this.displaySection(2)}}>
                    <EntityCoOcurrence
                        style={{display: this.displaySection(2)}}
                        dataset_id={dataset_id}
                        classifier_class={classifier_class}
                        data={predictions}
                        prev_data={prev_results}
                        show_predictions={show_predictions}
                        show_labels={show_labels}
                    />
                </div>
                <div style={{display: this.displaySection(3)}}>
                    <ResultVisualization
                        style={{display: this.displaySection(3)}}
                        dataset_id={dataset_id}
                        classifier_class={classifier_class}
                        show_predictions={show_predictions}
                        show_labels={show_labels}
                        data={{
                            predictions,
                            entire_data,
                            entry_to_sentences,
                            sentence_data,
                        }}
                    />
                </div>

                <MobileStepper
                    variant="dots"
                    steps={numberOfTrainIters}
                    position="static"
                    activeStep={this.state.trainsetStep}
                    className={classes.MobileStepper}
                    nextButton={
                    <Button size="small" onClick={this.handleNext.bind(this)} disabled={this.state.trainsetStep === (numberOfTrainIters - 1)}>
                        Next
                        {theme.direction === 'rtl' ? <KeyboardArrowLeft /> : <KeyboardArrowRight />}
                    </Button>
                    }
                    backButton={
                    <Button size="small" onClick={this.handleBack.bind(this)} disabled={this.state.trainsetStep === 0}>
                        {theme.direction === 'rtl' ? <KeyboardArrowRight /> : <KeyboardArrowLeft />}
                        Prev
                    </Button>
                    }
                />
                </Paper>
            </div>
        );
    }
}

DatasetEvaluation.propTypes = {
    classes: PropTypes.object.isRequired,
    theme: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    prediction_result: PropTypes.object,
};

export default withRoot(withStyles(styles, {withTheme: true})(DatasetEvaluation));