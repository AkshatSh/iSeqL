import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Tabs from '@material-ui/core/Tabs';
import CircularProgress from '@material-ui/core/CircularProgress';
import Tab from '@material-ui/core/Tab';
import withRoot from '../withRoot';
import DatasetSummary from '../components/datasetSummary';
import DatasetEvaluation from '../components/datasetEvaluation';
import DatasetTraining from '../components/datasetTraining';
import Cookies from 'universal-cookie';
import {COOKIE_LOGIN} from '../configuration';
import LoginButton from '../components/core/login_button';
import SaveModelButton from '../components/core/save_model';
import LoadModelButton from '../components/core/load_model';
import Progress from '../components/core/progress';
import TurkCheatsheet from '../turk/turk_cheatsheet';
import {is_valid, get_user_url} from '../utils';
import configuration from '../configuration';

function TabContainer(props) {
    const displayVal = props.display ? 'block' : 'none';
    return (
        <Typography component="div" style={{ padding: 8 * 3, display: displayVal}}>
        {props.children}
        </Typography>
    );
}

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
grow: {
    // flexGrow: 1,
},
tab : {
    // flexGrow: 30,
},
login_button: {
    marginLeft: '70%',
},
login_info: {
    textAlign: 'center',
    paddingTop: theme.spacing.unit * 20,
},
flexContainer: {
    display: 'inline-block',
    alignItems: 'center',
    boxSizing: 'border-box',
},
progress: {
    top: "50%",
    right: "50%",
    position: "fixed",
},
// content: {
//     top: theme.spacing.unit * 10,
// }
});

class Dataset extends React.Component {
    state = {
    
      // state params for the generic user
      open: false,
      selected_tab: 1,
      cookies: null,
      user_name: null,

      // state params for training
      training_progress: null,

      // state params for dataset
      dataset_prepared: false,
      running_prepare: false,
      predictions: null,
    };

    prepareDataset() {
        const {classifier_class, dataset_id} = this.props;
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
                dataset_prepared: true,
                running_prepare: false,
            });
        });
    }

    handleChange = (_, value) => {
            // this.setState({ selected_tab: value });
            this.setState({ selected_tab: value + 1 });
    }

    setUserName(user_name) {
        if (!this.state.dataset_prepared) {
            this.prepareDataset();
        }

        this.setState({
            user_name
        });
    }

    componentDidMount() {
        const cookies = new Cookies();
        const user_name = cookies.get(COOKIE_LOGIN);
        if (is_valid(user_name)) {
            this.prepareDataset();
        }
        this.setState({
            cookies,
            user_name,
        });
    }

    setProgress(progress_data) {
        let res = null;
        if (is_valid(progress_data)) {
            res = progress_data;
        }

        this.setState({appBarInfo: res});
    }

    createProgressBar() {
        const {appBarInfo} = this.state;
        if (!is_valid(appBarInfo)) {
            return null;
        }

        const {training_progress, component} = appBarInfo;
        if (is_valid(training_progress)) {
            const {progress_message, current, total} = training_progress;
            return (
                <Progress
                    progress_message={progress_message}
                    current={current}
                    total={total}
                />
            );
        } else if (is_valid(component)) {
            return component;
        }

        return null;
    }

    setPredictions(predictions) {
        this.setState({predictions});
    }

    setDatasetPreparedTrue() {
        // this.setState({dataset_prepared: true});
    }

    render() {
        const {theme, classes, classifier_class, dataset_id } = this.props;
        const {open, selected_tab, user_name, dataset_prepared, predictions} = this.state;
        const appBar = this.createProgressBar();
        let appBarStyle = {};
        if (selected_tab === 1) {
            appBarStyle = {position: "fixed"};
        }
        return (
        <div>
            <AppBar style={appBarStyle} position="static">
                <Tabs value={selected_tab - 1} onChange={this.handleChange}>
                    <Tab className={classes.tab} label="Labeling" />
                    {/* <Tab className={classes.tab} label="Summary" /> */}
                    <Tab className={classes.tab} label="Evaluation"/>
                    <span style={{flex: 1, textAlign: "center"}}>{appBar}</span> 
                    {/* <LoadModelButton /> */}
                    {/* <SaveModelButton /> */}
                    {/* <TurkCheatsheet /> */}
                    <LoginButton 
                        user_name={user_name}
                        handleSetUserName={this.setUserName.bind(this)}
                    />
                </Tabs>
            </AppBar>
            {user_name === null || user_name === undefined ? 
            <div className={classes.login_info}>
                <Typography variant="h4" gutterBottom>
                    Please Log In to Continue
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                </Typography>
                <LoginButton 
                    user_name={user_name}
                    handleSetUserName={this.setUserName.bind(this)}
                    loginProps={{variant: "contained"}}
                />
            </div>
            :
            dataset_prepared ? 
            <div className={classes.content}>
                {/* <TabContainer display={selected_tab === 0}>
                    <DatasetSummary
                        classifier_class={classifier_class}
                        dataset_id={dataset_id}
                        datasetPreparedFunc={this.setDatasetPreparedTrue.bind(this)}
                    />
                </TabContainer> */}
                <TabContainer display={selected_tab === 1}>
                    <DatasetTraining
                        classifier_class={classifier_class}
                        dataset_id={dataset_id}
                        updateProgressFunction={this.setProgress.bind(this)}
                        datasetPrepared={dataset_prepared}
                        setPredictionsCallback={this.setPredictions.bind(this)}
                    />
                </TabContainer>
                <TabContainer display={selected_tab === 2}>
                    <DatasetEvaluation 
                        classifier_class={classifier_class}
                        dataset_id={dataset_id}
                        prediction_result={predictions}
                    />
                </TabContainer>
            </div>
            :
            <CircularProgress className={classes.progress} color="secondary" />
            }
        </div>
        )
    }
}

Dataset.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
};

export default withRoot(withStyles(styles, {withTheme: true})(Dataset));