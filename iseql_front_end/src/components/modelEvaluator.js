import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import {sleep, get_user_url} from '../utils';

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

class ModelEvaluator extends React.Component {
    state = {
        running: false,
    };

    train_on_click() {
        this.setState({
            running: true,
        });

        const url = get_user_url(
            `${configuration.ACTIVE_LEARNING_SERVER}/api/train`,
            {},
        );
        fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(_ => { 
            this.setState({
                running: false,
            });
        });
    }

    render() {
        return (
            <div>
                <Typography variant="h4" gutterBottom>
                Model Training
                </Typography>
                <Button variant="contained" color="primary" onClick={this.train_on_click.bind(this)}>
                    {"Evaluate Model"}
                </Button>
            </div>
        );
    }
}

ModelEvaluator.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
};

export default withRoot(withStyles(styles)(ModelEvaluator));