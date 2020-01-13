import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import CircularProgress from '@material-ui/core/CircularProgress';
import configuration from '../configuration';
import withRoot from '../withRoot';
import dataset from '../pages/dataset';
import {get_user_url} from '../utils';

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

class DatasetSummary extends React.Component {
    state = {
        is_prepared: false,
        running_prepare: false,
    };

    componentDidMount() {
        this.handleClick();
    }


    handleClick() {
        const {classifier_class, dataset_id, datasetPreparedFunc} = this.props;
        this.setState({
            running_prepare: true,
        });

        const qparams = {
            session_id: dataset_id,
            ner_class: classifier_class,
        };

        const url = get_user_url(
            `${configuration.ACTIVE_LEARNING_SERVER}/api/set_session/`,
            qparams,
        );

        fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(_ => { 
            this.setState({
                is_prepared: true,
                running_prepare: false,
            });
            datasetPreparedFunc();
        });
    }

    render() {
        const {theme, classes, dataset_id, classifier_class} = this.props;
        const {is_prepared, running_prepare} = this.state;
        return (
            <div>
                <Typography variant="h4" gutterBottom>
                Dataset Summary
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                A tool to help build named entity and sequence labeling classifiers
                </Typography>
                {is_prepared ? 
                    <Button variant="contained" color="primary" disabled>
                        {"Already Prepared"}
                    </Button> 
                : 
                    running_prepare ? 
                    <CircularProgress className={classes.progress} color="secondary" />
                    :
                    <Button variant="contained" color="primary" onClick={this.handleClick.bind(this)}>
                        {"Prepare"}
                    </Button>
                }
            </div>
        );
    }
}

DatasetSummary.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    datasetPreparedFunc: PropTypes.func,
};

export default withRoot(withStyles(styles, {withTheme: true})(DatasetSummary));