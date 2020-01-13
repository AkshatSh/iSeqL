import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import Badge from '@material-ui/core/Badge';
import CircularProgress from '@material-ui/core/CircularProgress';
import { withStyles } from '@material-ui/core/styles';
import purple from '@material-ui/core/colors/purple';
// import styled from 'styled-components';
import ExampleLabeler from './exampleLabeler';
import ModelTrainer from './modelTrainer';
import withRoot from '../withRoot';
import {ACTIVE_LEARNING_SERVER, PROGRESS_WAIT_TIME} from '../configuration';
import SidePanel from './labeling/side_panel';
import TurkSuvery from '../turk/turk_survey';
import HistoricLabels from './labeling/historic_labels';
import {post_data, get_user_url, sleep, is_valid} from '../utils';
import progress from './core/progress';

const styles = theme => ({
root: {
    textAlign: 'center',
    paddingTop: theme.spacing.unit * 20,
},
margin: {
    margin: theme.spacing.unit * 2,
},
nested: {
    paddingLeft: theme.spacing.unit * 4,
},
fab: {
    margin: theme.spacing.unit,
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
container: {
    display: "flex",
},
main_panel: {
    flexGrow: 4,
},
side_panel: {
    flexGrow: 1,
    marginLeft: theme.spacing.unit * 2,
    width: theme.spacing.unit * 100,
},
paper_internal: {
    padding: theme.spacing.unit * 2,
    textAlign: "center",
    minWidth: theme.spacing.unit * 400,
},
updateModelButton: {
    margin: theme.spacing.unit,
},
textMessage: {
    margin: theme.spacing.unit * 2,
    fontFamily: 'Roboto',
    textAlign: 'center',
},
progressBackground: {
    position: "fixed",
    zIndex: 99999,
    top: "0",
    left: "0",
    bottom: "0",
    right: "0",
    background: "rgba(0,0,0,0.5)",
    transition: "1s 0.4s",
},
progressInternal: {
    height: 1,
    background: "#fff",
    position: "absolute",
    width: 0,
    top: "50%",
    right: "50%",
},
turk_survey: {
    marginTop: theme.spacing.unit * 10,
    width: '75%',
},
});

class DatasetTraining extends React.Component {
    state = {
        examples: [],
        example_indexes: [],
        example_predictions: {},
        refresh: true,
        kickoff_train: false,
        loading: false,
        experiment_finished: false,
        top_pred_ents: [],

        // state for training
        running_train: false,
        training_progress: {},
        training_thread_id: -1,

        // state for labeling
        label_iteration: 0,

        // for label progress
        flipped_data: [],
        labeled_set_sizes: [],
        train_summary: [],

        // prediction data
        predicted_data: {},

        newModel: null,

        // special case for final batch
        final_batch: false,
    };

    async componentDidMount() {
        await this.getPredictions(false);
        this.useAvailableModel();
        this.get_query(() => {});
    }

    get_query(on_done_func) {
        const url = get_user_url(
            `${ACTIVE_LEARNING_SERVER}/api/get_query/`,
            {},
        );
        fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(results => { 
            const {refresh} = this.state;
            const example_strs = [];
            const example_i = [];
            const example_predictions = results.predictions;
            const final_batch = results.final_batch;
            if (!is_valid(example_predictions)) {
                this.setState({experiment_finished: true});
                return; 
            }
            for (const i in results.results) {
                const result = results.results[i];
                example_i.push(result[0]);
                example_strs.push([result[0], result[1]]);
            }
            this.setState({
                examples: example_strs,
                example_indexes: example_i,
                refresh: !refresh,
                example_predictions,
                final_batch,
            });
            on_done_func();
            this.setExamplePredictions();
        });
    }

    async publish_training_examples(data = {}) {
        this.setState({loading: true});
        const url = get_user_url(
            `${ACTIVE_LEARNING_SERVER}/api/add_examples/`,
            {},
        );
        await post_data(url, {data}).then(() => {
            const {labeling_iteration} = this.state;
            if (!this.state.final_batch) {
                this.train_model();
            }
            this.setState({
                labeling_iteration: labeling_iteration + 1,
                loading: false,
            });
        });
    }

    train_model() {
        const {running_train} = this.state;
        const {updateProgressFunction} = this.props;
        // const {getPredictions} = this.props;
        if (running_train) {
            // already something running
            return;
        }

        this.setState({
            running_train: true,
            training_progress: {},
        });

        const url = get_user_url(
            `${ACTIVE_LEARNING_SERVER}/api/train`,
        );

        updateProgressFunction(
            {training_progress: {progress_message: "Starting Train...", total: 1, current: 0}}
        );

        fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(async (results) => {
            this.setState({
                training_thread_id: results.thread_id,
            });

            await this.checkTrainStatus();
            await this.getPredictions();
            this.setState({
                running_train: false
            });
        });
    }

    setPredictions(results) {
        this.setState({
            predicted_data: results.predicted_data,
            train_summary: results.training_summary,
            flipped_data: results.flipped_data,
            labeled_set_sizes: results.labeled_set_sizes,
            newModel: null,
        });
        this.setExamplePredictions(results.predicted_data);
        this.props.setPredictionsCallback(results);
    }

    setExamplePredictions(predicted_data=null) {
        if (!is_valid(predicted_data)) {
            predicted_data = this.state.predicted_data;
        }
        const {examples, example_predictions} = this.state;
        const pred_valid = is_valid(predicted_data) && Object.keys(predicted_data).length > 0;
        for (let i = 0; i < examples.length; i++) {
            const e_data = examples[i];
            const ei = e_data[0];
            const e_str = e_data[1];
            example_predictions[ei] = pred_valid ? 
                predicted_data[ei] :
                [e_str, null];
        }

        this.setState({example_predictions});
    }

    useAvailableModel() {
        const {updateProgressFunction} = this.props;
        this.setState({loading: true});
        if (is_valid(this.state.newModel)) {
            this.setPredictions(this.state.newModel);
        }

        updateProgressFunction(null);
        this.setState({loading: false});
    }

    setModelAvaiable(results, set_component=true) {
        const {classes, updateProgressFunction} = this.props;
        this.setState({
            newModel: results,
        });

        const component = <Badge color="error" badgeContent={"!"} className={classes.updateModelButton}>
            <Button
                variant="contained"
                style={{
                    webkitAnimation: "pulse 1.5s infinite",
                    boxShadow: "0 0 0 0 rgba(#5a99d4, .5)",
                }}
                onClick={this.useAvailableModel.bind(this)}>
                Update User Interface with New Model
            </Button>
        </Badge>;

        if (set_component) {
            updateProgressFunction({component});
        }
    }

    async getPredictions(set_component=true) {
        const {experiment_finished} = this.state;
        if (!set_component) {
            this.setState({loading: true});
        }
        const url = get_user_url(
            `${ACTIVE_LEARNING_SERVER}/api/predictions/`,
        );
        await fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(results => {
            this.setModelAvaiable(results, set_component);
            if (!set_component) {
                this.setState({loading: false});
            }

            if (this.experiment_finished) {
                // automaitcally update if the user has finished the experiment
                this.useAvailableModel();
            }
        });
    }

    async checkTrainStatus() {
        const {updateProgressFunction} = this.props;
        let {training_thread_id} = this.state;
        const base_url = `${ACTIVE_LEARNING_SERVER}/api/progress/`; 
        let wait_time = 0;
        updateProgressFunction(this.updateTrainingProgress());
        while (training_thread_id > 0) {

            // wait 10 seconds before checking for progress
            await sleep(wait_time);
            const qparams = {thread_id: training_thread_id};
            const url = get_user_url(base_url, qparams);
            await fetch(
                url,
            ).then(results => {
                return results.json();
            }).then(results => {
                if ('Error' in results) {
                    training_thread_id = -1;
                } else {
                    this.setState({
                        training_progress: results,
                    });
                    updateProgressFunction(this.updateTrainingProgress());
                }
            });

            // wait 10 seconds before checking again
            wait_time = PROGRESS_WAIT_TIME;
        }

        await fetch(
            get_user_url(`${ACTIVE_LEARNING_SERVER}/api/trainer_progress/`)
        ).then(results => {
            return results.json();
        }).then(results => {
            this.setState({
                training_progress: {train_progress: results},
            });
            updateProgressFunction(this.updateTrainingProgress());
        });


        updateProgressFunction(null);
        this.setState({
            training_thread_id,
        });
    }

    updateTrainingProgress() {
        const {classes} = this.props;
        const {running_train, training_progress} = this.state;
        const training_progress_data = training_progress.train_progress;
        const total_epochs = training_progress.num_epochs;

        let progress_message = null;
        if (is_valid(training_progress_data)) {
            progress_message = `Training... ${training_progress_data.length}/${total_epochs}`;
            if (training_progress_data.length >= total_epochs) {
                progress_message = "Evaluating Model On All Data ...";
            }
        }

        const total = total_epochs + 1;
        const current = (is_valid(training_progress_data) ? training_progress_data.length: 0); 
        return is_valid(progress_message) ? 
            {training_progress: {progress_message, total, current}} : null;
    }

    loading_div() {
        const {classes} = this.props;
        return <div className={classes.progressBackground}>
            <div className={classes.progressInternal}>
                <CircularProgress className={classes.progress} color="secondary" />
            </div>
        </div>
    }

    updateTopEntities(ents) {
        // console.log(ents.slice(0, 5));
        this.setState({top_pred_ents: ents});
    }

    load_survey() {
        const {classes} = this.props;
        const {running_train, top_pred_ents, newModel} = this.state;
        const internal = running_train || newModel !== null ? 
            <Typography variant='h2'>
                Wait for train to finish...
            </Typography>
            :
            <TurkSuvery top_entities={top_pred_ents} />;

        return <Paper className={classes.turk_survey}>{internal}</Paper>;
    }

    render() {
        const {classes, dataset_id, classifier_class} = this.props;
        const {
            loading,
            examples,
            example_predictions,
            refresh,
            flipped_data,
            train_summary,
            labeled_set_sizes,
            predicted_data,
            running_train,
            experiment_finished,
        } = this.state;

        return (
            <div className={classes.container}>
                {loading ? this.loading_div() : null}
                <div className={classes.main_panel}>
                    {experiment_finished ? this.load_survey() : 
                    <ExampleLabeler
                        className={classes.paper}
                        key={"example_labeler_" + refresh}
                        classifier_class={classifier_class}
                        dataset_id={dataset_id}
                        examples={examples}
                        add_training_examples_func={this.publish_training_examples.bind(this)}
                        fetch_query_func={this.get_query.bind(this)}
                        example_predictions={example_predictions}
                        disable_publish={running_train}
                        predicted_data={predicted_data}
                    />
                    }
                </div>
                <SidePanel
                    header={
                        <Typography className={classes.updateModelButton} style={{backgroundColor: purple[200], margin: 0}} variant="h6">Model Results</Typography>
                    }
                    style={{width: '300px'}}
                    className={classes.side_panel}
                    predictions={predicted_data}
                    train_summary={train_summary}
                    labeled_set_sizes={labeled_set_sizes}
                    flipped_data={flipped_data}
                    updateTopEntities={this.updateTopEntities.bind(this)}
                />
            </div>
        );
    }
}

DatasetTraining.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    updateProgressFunction: PropTypes.func,
    dataset_prepared: PropTypes.bool,
    getPredictions: PropTypes.func,
    setPredictionsCallback: PropTypes.func,
};

export default withRoot(withStyles(styles, {withTheme: true})(DatasetTraining));