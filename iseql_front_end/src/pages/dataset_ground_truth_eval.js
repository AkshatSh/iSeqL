import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import Button from '@material-ui/core/Button';
import JSONInput from 'react-json-editor-ajrm';
import ReactJson from 'react-json-view'
import AppBar from '@material-ui/core/AppBar';
import locale from 'react-json-editor-ajrm/locale/en';
import {ACTIVE_LEARNING_SERVER} from '../configuration';
import {get_user_url} from '../utils';
import Cookies from 'universal-cookie';
import {COOKIE_LOGIN} from '../configuration';
import LoginButton from '../components/core/login_button';

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

class DatasetGroundTruth extends React.Component {

    state = {
        ground_truth: {},
        user_name: null,
    }

    evaluateClick() {
        const url = get_user_url(
            `${ACTIVE_LEARNING_SERVER}/api/compute_ground_truth`,
        );
        fetch(
            url,
        ).then(results => {
            return results.json();
        }).then(results => { 
            this.setState({ground_truth: results});
        });
    }

    setUserName(user_name) {
        this.setState({
            user_name
        });
    }

    componentDidMount() {
        const cookies = new Cookies();
        const user_name = cookies.get(COOKIE_LOGIN);
        this.setState({
            cookies,
            user_name,
        });
    }

    render() {
        const {classes} = this.props;
        const {user_name, ground_truth} = this.state;
        return (
            <div>
                <AppBar position="static">
                    <span style={{flex: 1}} /> 
                    <LoginButton 
                        user_name={user_name}
                        handleSetUserName={this.setUserName.bind(this)}
                    />
                </AppBar>
                <Button
                    variant="contained"
                    color="primary"
                    className={classes.fab}
                    onClick={this.evaluateClick.bind(this)}
                    >
                    Get Ground Truth Result
                </Button>
                <ReactJson src={ground_truth} />
            </div>
        );
    }
}

DatasetGroundTruth.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
};

export default withRoot(withStyles(styles)(DatasetGroundTruth));