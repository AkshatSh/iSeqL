import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';

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

class DatasetExplorer extends React.Component {
    state = {
        is_prepared: false,
    };

    render() {
        const {classes, dataset_id, classifier_class} = this.props;
        const {is_prepared} = this.state;
        return (
            <div>
                <Typography variant="h4" gutterBottom>
                DATASET SUMMARY
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                A tool to help build named entity and sequence labeling classifiers
                </Typography>
                <Button variant="contained" color="primary" onClick={this.handleClick.bind(this)}>
                    {"Explore Data"}
                </Button>
            </div>
        );
    }
}

DatasetExplorer.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
};

export default withRoot(withStyles(styles)(DatasetExplorer));