import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import ResultTable from './resultTable';

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
});

class ResultExplorer extends React.Component {

    render() {
        const {data, dataset_id, classifier_class, is_predicted, combined_view, show_predicted, show_labeled} = this.props;
        return (
            <div>
                <ResultTable
                    dataset_id={dataset_id}
                    classifier_class={classifier_class}
                    data={data}
                    is_predicted={is_predicted}
                    combined_view={combined_view}
                    show_predicted={show_predicted}
                    show_labeled={show_labeled}
                />
            </div>
        );
    }
}

ResultExplorer.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object,
    is_predicted: PropTypes.bool,
    combined_view: PropTypes.bool,
    show_predicted: PropTypes.bool,
    show_labeled: PropTypes.bool,
};

export default withRoot(withStyles(styles)(ResultExplorer));