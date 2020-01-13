import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import LinearProgress from '@material-ui/core/LinearProgress';
import DialogTitle from '@material-ui/core/DialogTitle';
import Dialog from '@material-ui/core/Dialog';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import CircularProgress from '@material-ui/core/CircularProgress';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import IconButton from '@material-ui/core/IconButton';
import CloseIcon from '@material-ui/icons/Close';
import Publish from '@material-ui/icons/Publish';
import VerticalAlignBottom from '@material-ui/icons/VerticalAlignBottom';
import FormControl from '@material-ui/core/FormControl';
import FormGroup from '@material-ui/core/FormGroup';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import Switch from '@material-ui/core/Switch';
import Slide from '@material-ui/core/Slide';
import shallowCompare from 'react-addons-shallow-compare'; 
import HistoricLabels from './labeling/historic_labels';
import withRoot from '../withRoot';
import ExampleCard from './exampleCard';
import {TURK_CONDITION_SHOW_PREDICTIONS} from '../configuration';
import { is_valid } from '../utils';

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
    hidden: {
        display: "none",
    },
    show_predictions: false,
    empty : {},
    paper: {
        ...theme.mixins.gutters(),
        paddingTop: theme.spacing.unit * 2,
        paddingBottom: theme.spacing.unit * 2,
        marginTop: theme.spacing.unit * 2,
    },
    progressBar: {
        flexGrow: 1,
        marginBottom: theme.spacing.unit * 1,
        marginTop: theme.spacing.unit * 1
    },
    panel: {
        top: theme.spacing.unit * 7,
        width: '75%',
    },
    dataset_dialog: {
        padding: theme.spacing.unit * 4,
    },
    appBar: {
        position: 'relative',
    },
});

function Transition(props) {
    return <Slide direction="up" {...props} />;
}

class ExampleLabeler extends React.Component {
    state = {
        examples: [],
        labeled_examples: {},
        labeled_indexes: {},
        show_predictions: TURK_CONDITION_SHOW_PREDICTIONS,
        expanded: 'panel1',
        fetching_query: false,
        open_view_dataset: false,
    };

    shouldComponentUpdate(nextProps, nextState) {
        const nextStateTemp = {...nextState};
        nextStateTemp.labeled_examples = this.state.labeled_examples;
        nextStateTemp.labeled_indexes = this.state.labeled_indexes;

        const testA = shallowCompare(this, nextProps, nextStateTemp);
        // if (testA) {
        //     return true;
        // }
        // const {example_predictions} = this.props;
        // if (is_valid(example_predictions) !== is_valid(nextProps.example_predictions)) {
        //     return true;
        // }

        // const eids = Object.keys(example_predictions);
        // if (eids !== Object.keys(nextProps.example_predictions)) {
        //     return true;
        // }

        // if (example_predictions[eids[0]] !== nextProps.example_predictions[eids[0]]) {
        //     return true;
        // }

        // if (example_predictions[eids[0]][1].length !== nextProps.example_predictions[eids[0]][1].length) {
        //     return true;
        // }

        // if (example_predictions[eids[0]][1] !== nextProps.example_predictions[eids[0]][1]) {
        //     return true;
        // }


        return testA;
    }

    doneLabelFunc(index, example, ranges) {
        const {labeled_indexes, labeled_examples, examples} = this.state;
        labeled_indexes[parseInt(index)] = true;
        const pos_label = is_valid(ranges) ? [] : null;
        for (var i in ranges) {
            const range = ranges[i];
            // + 1 for exclusive
            pos_label.push([range.word_start, range.word_end + 1]);
        }
        labeled_examples[parseInt(index)] = [example, pos_label];
        this.setState({
            labeled_indexes: labeled_indexes,
            labeled_examples: labeled_examples,
        });

        // if (Object.keys(labeled_indexes).length === examples.length) {
        //     this.send_training_examples();
        // }
    }

    componentDidMount() {
        const {examples} = this.props;
        const labeled_indexes = {};
        const labeled_examples = {};

        for (const ei in examples) {
            const example_data = examples[ei];
            const example_id = example_data[0];
            const example_str = example_data[1];
            labeled_indexes[parseInt(example_id)] = true;
            labeled_examples[parseInt(example_id)] = [example_str, []];
        }

        this.setState({
            examples,
            labeled_indexes,
            labeled_examples,
        });
    }

    async send_training_examples() {
        const {add_training_examples_func} = this.props;
        const {labeled_examples, label_iteration} = this.state;
        await add_training_examples_func(
            labeled_examples,
        );

        // reset the examples
        this.setState({
            examples: [],
            labeled_examples: {},
            labeled_indexes: {},
        });

        this.onGetMoreClick();
    }

    handleChange = name => event => {
        this.setState({ ...this.state, [name]: event.target.checked });
    };

    handlePanelChange = panel => (event, isExpanded) => {
        this.setState({expanded:  isExpanded ? panel : false});
    };

    onGetMoreClick() {
        const {fetch_query_func} = this.props;
        this.setState({
            fetching_query: true,
        });

        fetch_query_func(this.finishedFetching.bind(this));

    }

    finishedFetching() {
        this.setState({
            fetching_query: false,
        });
    }

    view_dataset() {
        this.setState({
            open_view_dataset: true,
        });
    }

    close_dataset() {
        this.setState({
            open_view_dataset: false,
        });
    }

    create_dialog() {
        const {classes, classifier_class, dataset_id, predicted_data} = this.props;
        return <Dialog
            fullScreen
            onClose={this.close_dataset.bind(this)}
            aria-labelledby="dataset-dialog"
            TransitionComponent={Transition}
            open={this.state.open_view_dataset}>
            <AppBar className={classes.appBar}>
                <Toolbar>
                    <IconButton color="inherit" onClick={this.close_dataset.bind(this)} aria-label="Close">
                        <CloseIcon />
                    </IconButton>
                    <Typography style={{textAlign: "center", flexGrow: 1, color: "white"}} variant="h5">
                        Dataset Explorer
                    </Typography>
                </Toolbar>
            </AppBar>
            <div className={classes.dataset_dialog}>
            <HistoricLabels
                classifier_class={classifier_class}
                dataset_id={dataset_id}
                predictions={predicted_data}
            />
            </div>
        </Dialog>;
    }

    render() {
        const {classes, dataset_id, classifier_class, example_predictions, disable_publish, theme} = this.props;
        const {expanded, examples, show_predictions, fetching_query} = this.state;
        // const example_cards = [];
        let num_labeled = 0.0;
        let total = 0.0;

        const example_cards = examples.map((example_data) => {
            const example_id = example_data[0];
            const example = example_data[1];

            let example_prediction = example_predictions[example_id];
            if (example_prediction[1] !== null) {
                example_prediction = example_prediction[1];
            } else {
                example_prediction = null;
            }

            return (
                <div key={"div_example_card_" + example_id}>
                    <ExampleCard
                        key={"example_card_" + example_id}
                        classifier_class={classifier_class}
                        dataset_id={dataset_id}
                        example={example}
                        doneLabelFunc={this.doneLabelFunc.bind(this)}
                        index={example_id}
                        example_prediction={example_prediction}
                        show_predictions={show_predictions}
                    />
                </div>
            );
        });

        const publishProps = {disabled: disable_publish};
        const buttonFontSize = 20;

        return (
            <ExpansionPanel
                expanded={expanded === 'panel1'}
                onChange={this.handlePanelChange('panel1').bind(this)}
                className={classes.panel}
            >
                <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography style={{margin: 0}} variant="h4" gutterBottom>Label Examples</Typography>
                </ExpansionPanelSummary>
                {this.create_dialog()}
                <ExpansionPanelDetails style={{display: 'block'}}>
                    <div>
                        <Button
                            variant="outlined"
                            color="primary"
                            onClick={this.view_dataset.bind(this)}>
                            {"View Dataset"}
                        </Button>
                    </div>
                    <div>
                    {disable_publish ?
                    <div>
                        <Typography variant='body1'>You can continue to label as the model trains</Typography>
                    </div>
                    : null}
                    {example_cards.length > 0 ?
                    <div>
                        <div>
                            <FormControl component="fieldset">
                                {/* <Button
                                    style={{marginRight: theme.spacing.unit}}
                                    variant="outlined"
                                    color="primary"
                                    onClick={this.view_dataset.bind(this)}>
                                    {"View Dataset"}
                                </Button> */}
                                {/* <FormGroup style ={{display: "inline"}}>
                                    <FormControlLabel
                                        control={
                                        <Switch
                                            checked={show_predictions}
                                            onChange={this.handleChange('show_predictions').bind(this)}
                                            value="show_predictions"
                                            color="secondary"
                                        />
                                        }
                                        label="Show Model Predictions"
                                    />
                                </FormGroup> */}
                            </FormControl>
                            {/* <LinearProgress
                                className={classes.progressBar}
                                color="secondary"
                                variant="determinate"
                                value={num_labeled / total * 100} /> */}
                            {example_cards}
                            <Button 
                                    {...publishProps}
                                    variant="contained"
                                    color="primary"
                                    onClick={this.send_training_examples.bind(this)}>
                                <Publish style={{marginRight: 5, fontSize: buttonFontSize}} />
                                {"Publish Examples"}
                            </Button>
                            {disable_publish ? <Typography variant='body1' color='error'>Please Wait for Training to Finish</Typography> : null}
                        </div>
                    </div>
                    : 
                        fetching_query ?
                            <CircularProgress color="primary" />
                        :
                            <div>
                                {/* <Button
                                    style={{marginRight: theme.spacing.unit}}
                                    variant="outlined"
                                    color="primary"
                                    onClick={this.view_dataset.bind(this)}>
                                    {"View Dataset"}
                                </Button> */}
                                <Button
                                    style={{marginTop: 10}}
                                    variant="contained"
                                    color="primary"
                                    onClick={this.onGetMoreClick.bind(this)}>
                                    <VerticalAlignBottom style={{marginRight: 5, fontSize: buttonFontSize}} />
                                    {"Get More Examples"}
                                </Button> 
                            </div>
                    }
                    </div>
                </ExpansionPanelDetails>
            </ExpansionPanel>
        );
    }
}

ExampleLabeler.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    examples: PropTypes.array,
    example_predictions: PropTypes.object,
    add_training_examples_func: PropTypes.func,
    fetch_query_func: PropTypes.func,
    disable_publish: PropTypes.bool,
    predicted_data: PropTypes.object,
};

export default withRoot(withStyles(styles, {withTheme: true})(ExampleLabeler));