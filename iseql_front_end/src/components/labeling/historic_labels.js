import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import Paper from '@material-ui/core/Paper';
import withRoot from '../../withRoot';
import ResultExplorer from '../resultExplorer';
import ResultTable from '../resultTable';
import ResultControls from '../resultControls';
import {getLast} from '../../utils';

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
panel: {
    top: theme.spacing.unit * 7,
    width: '75%',
}
});

class HistoricLabels extends React.Component {
    state = {
        predictions: {},
        training_set_sizes: [],
        train: true,
        test: true,
        unlabeled: true,
        combined_view: false,
        show_predictions: false,
        expanded: null,
        show_labels: true,
        show_predictions: false,
    };

    handlePanelChange = panel => (event, isExpanded) => {
        this.setState({expanded:  isExpanded ? panel : false});
    };

    onControlUpdate(name, value) {
        this.setState({ ...this.state, [name]: value });
    }

    filterResults() {
        const {predictions} = this.props;
        const {train, test, unlabeled} = this.state;
        const result = {};
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
            new_entry_data[1].entities = getLast(new_entry_data[1].entities, []); //[trainsetStep];
            new_entry_data[1].ranges = getLast(new_entry_data[1].ranges, []); // [trainsetStep];
            result[s_id] = new_entry_data;
        }

        return {result};
    }

    render() {
        const {classes, dataset_id, classifier_class} = this.props;
        const {show_predictions, expanded, show_labels} = this.state;
        const {result} = this.filterResults();
        const predictions = result;
        return (
            <div style={{width: "100%"}}>
                <ResultControls
                    onControlsUpdate={this.onControlUpdate.bind(this)}
                    fetchPredictionsFunc={null}
                />
                {/* <ResultExplorer
                    dataset_id={dataset_id}
                    classifier_class={classifier_class}
                    data={predictions}
                    is_predicted={show_predictions}
                    combined_view={show_predictions && show_labels}
                    show_predicted={show_predictions}
                    show_labeled={show_labels}
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
        );
    }
}

HistoricLabels.propTypes = {
    classes: PropTypes.object.isRequired,
    theme: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    predictions: PropTypes.object,
};

export default withRoot(withStyles(styles, {withTheme: true})(HistoricLabels));